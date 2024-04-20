import os
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, TrainerCallback, DataCollatorWithPadding, GenerationConfig
import torch
import logging
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import random
import wandb
from typing import Dict, List
import torch.nn.functional as F
import sys

def preprocess_function(source_text, target_text, tokenizer):
    prefix_text = f"Below is an instruction that describes a task. Write a response that appropriately completes the request. ### Instruction:{source_text} ### Response:"
    response_text = f"{prefix_text}{target_text}</s>"
    
    input = tokenizer(response_text, return_tensors=None,  max_length=128, truncation=True, padding='max_length')
    input_ids = input['input_ids']

    labels = input_ids.copy()
    
    output_start_index = len(tokenizer.encode(prefix_text,max_length=128, truncation=True, padding='max_length')) - 9
    labels[:output_start_index] = [-100] * output_start_index

    return {
        'input_ids': input_ids,
        'attention_mask': input['attention_mask'],
        'labels': labels
    }


def prefix_allowed_tokens_fn(candidate_trie):
    def prefix_allowed_tokens(batch_id, sentence):
        sentence = sentence.tolist()
        trie_out = candidate_trie.get(sentence)
        return trie_out

    return prefix_allowed_tokens


def llama_prefix_allowed_tokens_fn(candidate_trie):
    def prefix_allowed_tokens(batch_id, sentence):
        sentence = sentence.tolist()
        # 13291 is the token id for vokens, so the function can return the correct allowed tokens
        # you may need to change this value in your case
        index = sentence.index(13291) 
        sentence = sentence[index:]
        trie_out = candidate_trie.get(sentence)
        return trie_out

    return prefix_allowed_tokens


def load_codes(target_file):
    with  open(target_file, 'r', encoding='utf-8') as tgt_file:
        target_lines = tgt_file.readlines()
    res = []
    for i in range(len(target_lines)):
        if i % 5 == 0 and target_lines[i].strip() not in res:
            res.append(target_lines[i].strip())
    target_lines = res   

    return target_lines


def load_prompt(source_file,target_file,sub_size=None):
    target_lines = open(target_file, encoding='utf-8').read().splitlines()
    source_lines = open(source_file, encoding='utf-8').read().splitlines()
    
    if sub_size is not None and sub_size < len(source_lines):
        indeces = np.random.choice(len(source_lines), sub_size, replace=False)
        source_lines = [source_lines[i] for i in indeces]
        target_lines = [target_lines[i] for i in indeces]
    
    source = []
    target = []
    for i in range(len(source_lines)):
        source_text = source_lines[i]
        target_text = target_lines[i]
        prefix_text = f"Below is an instruction that describes a task. Write a response that appropriately completes the request. ### Instruction:{source_text} ### Response:"
        source.append(prefix_text)
        target.append(target_text)
    
    return source, target


def load_response(source_file,target_file):
    target_lines = open(target_file, encoding='utf-8').read().splitlines()
    source_lines = open(source_file, encoding='utf-8').read().splitlines()
    
    res = []
    for i in range(len(source_lines)):
        prefix_text = f"Response:{target_lines[i]}</s>"
        if prefix_text not in res:
            res.append(prefix_text)

    return res


class Trie(object):
    def __init__(self, sequences: List[List[int]] = []):
        self.trie_dict = {}
        self.len = 0
        if sequences:
            for sequence in sequences:
                Trie._add_to_trie(sequence, self.trie_dict)
                self.len += 1

        self.append_trie = None
        self.bos_token_id = None

    def append(self, trie, bos_token_id):
        self.append_trie = trie
        self.bos_token_id = bos_token_id

    def add(self, sequence: List[int]):
        Trie._add_to_trie(sequence, self.trie_dict)
        self.len += 1

    def get(self, prefix_sequence: List[int]):
        return Trie._get_from_trie(
            prefix_sequence, self.trie_dict, self.append_trie, self.bos_token_id
        )

    @staticmethod
    def load_from_dict(trie_dict):
        trie = Trie()
        trie.trie_dict = trie_dict
        trie.len = sum(1 for _ in trie)
        return trie

    @staticmethod
    def _add_to_trie(sequence: List[int], trie_dict: Dict):
        if sequence:
            if sequence[0] not in trie_dict:
                trie_dict[sequence[0]] = {}
            Trie._add_to_trie(sequence[1:], trie_dict[sequence[0]])

    @staticmethod
    def _get_from_trie(
        prefix_sequence: List[int],
        trie_dict: Dict,
        append_trie=None,
        bos_token_id: int = None,
    ):
        if len(prefix_sequence) == 0:
            output = list(trie_dict.keys())
            if append_trie and bos_token_id in output:
                output.remove(bos_token_id)
                output += list(append_trie.trie_dict.keys())
            return output
        elif prefix_sequence[0] in trie_dict:
            return Trie._get_from_trie(
                prefix_sequence[1:],
                trie_dict[prefix_sequence[0]],
                append_trie,
                bos_token_id,
            )
        else:
            if append_trie:
                return append_trie.get(prefix_sequence)
            else:
                return []

    def __iter__(self):
        def _traverse(prefix_sequence, trie_dict):
            if trie_dict:
                for next_token in trie_dict:
                    yield from _traverse(
                        prefix_sequence + [next_token], trie_dict[next_token]
                    )
            else:
                yield prefix_sequence

        return _traverse([], self.trie_dict)

    def __len__(self):
        return self.len

    def __getitem__(self, value):
        return self.get(value)


class T5Dataset(Dataset):
    def __init__(self, tokenizer, source_file, target_file, max_source_len=128, max_target_len=8, add_prefix=False,subset_size=None):
        self.tokenizer = tokenizer
        self.source_texts = open(source_file, encoding='utf-8').read().splitlines()
        self.target_texts = open(target_file, encoding='utf-8').read().splitlines()
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len
        self.subset_size = subset_size  
        self.add_prefix = add_prefix

        if self.subset_size is not None:
            indices = list(range(len(self.source_texts)))
            sampled_indices = random.sample(indices, self.subset_size)
            self.source_texts = [self.source_texts[i] for i in sampled_indices]
            self.target_texts = [self.target_texts[i] for i in sampled_indices]

    def __len__(self):
        return len(self.source_texts)

    def __getitem__(self, idx):
        source_text = self.source_texts[idx]
        if self.add_prefix:
            source_text = f"Below is an instruction that describes a task. Write a response that appropriately completes the request. Instruction:{source_text} Response:"
        target_text = self.target_texts[idx]

        source_encodings = self.tokenizer(source_text, padding='max_length', max_length=self.max_source_len, truncation=True, return_tensors="pt")
        target_encodings = self.tokenizer(target_text, padding='max_length', max_length=self.max_target_len, truncation=True, return_tensors="pt")
        #print(target_encodings)

        input_ids = source_encodings["input_ids"]
        labels = target_encodings["input_ids"]
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {"input_ids": input_ids.squeeze(), "labels": labels.squeeze()}
        # return  {"input_ids": source_text, "labels": target_text}


class LLaMaDataset(Dataset):
    def __init__(self, tokenizer, source_file, target_file, subset_size=None):
        self.tokenizer = tokenizer
        self.source_texts = open(source_file, encoding='utf-8').read().splitlines()
        self.target_texts = open(target_file, encoding='utf-8').read().splitlines()
        self.subset_size = subset_size  

        if self.subset_size is not None:
            indices = list(range(len(self.source_texts)))
            sampled_indices = random.sample(indices, self.subset_size)
            self.source_texts = [self.source_texts[i] for i in sampled_indices]
            self.target_texts = [self.target_texts[i] for i in sampled_indices]

    def __len__(self):
        return len(self.source_texts)

    def __getitem__(self, idx):
        source_text = self.source_texts[idx]
        target_text = self.target_texts[idx]
        return preprocess_function(source_text, target_text, self.tokenizer)


class QueryEvalCallback(TrainerCallback):
    def __init__(self, local_rank,test_dataset_1, test_dataset_2, tgt_file,logger, batch_size, collator, tokenizer,wandb=None,log_freq=3,gen_len=20):
        self.tokenizer = tokenizer
        self.logger = logger
        self.test_dataset_1 = test_dataset_1
        self.test_dataset_2 = test_dataset_2
        self.dataloader_1 = DataLoader(
            test_dataset_1,
            batch_size=batch_size,
            collate_fn=collator,
            shuffle=True,
            drop_last=False,
            num_workers=10,
        )
        self.dataloader_2 = DataLoader(
            test_dataset_2,
            batch_size=batch_size,
            collate_fn=collator,
            shuffle=True,
            drop_last=False,
            num_workers=10,
        )
        self.code_list = load_codes(tgt_file)
        condidate_trie = Trie([[0]+self.tokenizer.encode(x) for x in self.code_list])
        #print(condidate_trie)
        self.test_prefix_allowed_tokens_fn = prefix_allowed_tokens_fn(condidate_trie)
        self.wandb = wandb if wandb else None
        self.log_freq = log_freq
        self.gen_len = gen_len
        self.local_rank = local_rank

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.local_rank != 0:
            return
        current_epoch = state.epoch
        if int(current_epoch) % self.log_freq != 0:
            return
        recall_count_at_1_1 = 0
        recall_count_at_5_1 = 0
        recall_count_at_10_1 = 0
        recall_count_at_1_2 = 0
        recall_count_at_5_2 = 0
        recall_count_at_10_2 = 0
        model = kwargs['model'].eval()
        
        for batch_1 in tqdm(self.dataloader_1, desc='Evaluating train queries'):
            inputs_1 = batch_1
            with torch.no_grad():
                generation_config = GenerationConfig(
                        num_beams=10,
                        max_new_tokens=self.gen_len,
                        num_return_sequences=10,
                        # output_scores=True,
                        # return_dict_in_generate=True,
                        early_stopping=True,
                        use_cache=False,
                    )
                batch_beams_1 = model.generate(inputs_1['input_ids'].to(model.device),
                                        generation_config=generation_config
                                        ).reshape(inputs_1['input_ids'].shape[0], 10, -1)
                for beams, label in zip(batch_beams_1, inputs_1['labels']):
                    rank_list = self.tokenizer.batch_decode(beams, skip_special_tokens=True) 
                    rank_list = [x.split(' ') for x in rank_list]
                    label[label == -100] = self.tokenizer.pad_token_id
                    label = self.tokenizer.decode(label, skip_special_tokens=True).strip()
                    label = label.split(' ')
                    hits = [i for i, x in enumerate(rank_list) if x == label]
                    hits = [x for x in hits if x < 10]
                    if len(hits) != 0:
                        recall_count_at_10_1 += 1
                        if hits[0] < 5:
                            recall_count_at_5_1 += 1
                        if hits[0] == 0:
                            recall_count_at_1_1 += 1
        
        hits_at_1_data_1 = recall_count_at_1_1 / len(self.test_dataset_1)
        hits_at_5_data_1 = recall_count_at_5_1 / len(self.test_dataset_1)
        hits_at_10_data_1 = recall_count_at_10_1 / len(self.test_dataset_1)
        
        log_msg = f"Epoch {current_epoch} training set: Recall@1: {hits_at_1_data_1}, Recall@5: {hits_at_5_data_1}, Recall@10: {hits_at_10_data_1}"
        self.logger.info(log_msg)
        if self.wandb:
            self.wandb.log({' Train Recall@1': hits_at_1_data_1, ' Train Recall@5': hits_at_5_data_1, ' Train Recall@10': hits_at_10_data_1})

        for batch_2 in tqdm(self.dataloader_2, desc='Evaluating test queries'):
            inputs_2 = batch_2
            with torch.no_grad():
                generation_config = GenerationConfig(
                        num_beams=10,
                        max_new_tokens=self.gen_len,
                        num_return_sequences=10,
                        # output_scores=True,
                        # return_dict_in_generate=True,
                        early_stopping=True,
                        use_cache=False,
                    )
                batch_beams_2 = model.generate(inputs_2['input_ids'].to(model.device),
                                        generation_config=generation_config,
                                        prefix_allowed_tokens_fn=self.test_prefix_allowed_tokens_fn
                                        ).reshape(inputs_2['input_ids'].shape[0], 10, -1)
                
                for beams, label in zip(batch_beams_2, inputs_2['labels']):
                    rank_list = self.tokenizer.batch_decode(beams, skip_special_tokens=True) 
                    rank_list = [x.split(' ') for x in rank_list]
                    label[label == -100] = self.tokenizer.pad_token_id
                    label = self.tokenizer.decode(label, skip_special_tokens=True).strip()
                    label = label.split(' ')

                    hits = [i for i, x in enumerate(rank_list) if x == label]
                    hits = [x for x in hits if x < 10]
                    # print(rank_list)
                    # print(label)
                    # print(hits)
                    if len(hits) != 0:
                        recall_count_at_10_2 += 1
                        if hits[0] < 5:
                            recall_count_at_5_2 += 1
                        if hits[0] == 0:
                            recall_count_at_1_2 += 1

        hits_at_1_data_2 = recall_count_at_1_2 / len(self.test_dataset_2)
        hits_at_5_data_2 = recall_count_at_5_2 / len(self.test_dataset_2)
        hits_at_10_data_2 = recall_count_at_10_2 / len(self.test_dataset_2)

        log_msg = f"Epoch {current_epoch} test set: Recall@1: {hits_at_1_data_2}, Recall@5: {hits_at_5_data_2}, Recall@10: {hits_at_10_data_2}\n"
        self.logger.info(log_msg)
        if self.wandb:
            self.wandb.log({' Test Recall@1': hits_at_1_data_2, ' Test Recall@5': hits_at_5_data_2, ' Test Recall@10': hits_at_10_data_2})


class TrainerwithTemperature(Trainer):
    def __init__(self, temperature=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = temperature

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        logits = logits / self.temperature
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        # print(logits.shape, labels.shape)
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        return (loss, outputs) if return_outputs else loss
    

class LlaMaTrainerwithTemperature(Trainer):
    def __init__(self, temperature=1.0, vocab_size=32000,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = temperature
        self.vocab_size = vocab_size

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        logits = logits / self.temperature

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        shift_logits = shift_logits.view(-1, self.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

        return (loss, outputs) if return_outputs else loss


class LTRTrainer(Trainer):
    def __init__(self, temperature=1.0, ltr_loss_factor=1.0, train_allowed_tokens=None, margin=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = temperature
        self.train_allowed_tokens = train_allowed_tokens
        self.ltr_loss_factor = ltr_loss_factor
        self.margin = margin

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        logits = logits / self.temperature

        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        tem_loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        ltr_loss = self.multi_ltr_loss(model, inputs, logits, labels)
        
        loss = self.ltr_loss_factor * ltr_loss + tem_loss

        return (loss, outputs) if return_outputs else loss
    

    def ltr_loss(self, model, inputs, logits, labels, margin):

        if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
            model = model.module

        num_beams = 10
        generation_config = GenerationConfig(
            num_beams=num_beams,
            max_new_tokens=10,
            num_return_sequences=num_beams,
            early_stopping=True,
            use_cache=False,
        )

        beams = model.generate(inputs.get("input_ids"),
                                generation_config=generation_config,
                                prefix_allowed_tokens_fn=self.train_allowed_tokens
                                ).reshape(inputs.get("input_ids").shape[0], 10, -1)
        
        # positive scores
        positive_logits = logits[:,:5,:]
        positive_probs = F.log_softmax(positive_logits, dim=-1)
        positive_selected_probs = torch.gather(positive_probs, 2, labels[:,:5].unsqueeze(-1)).squeeze(-1)
        positive_scores = positive_selected_probs.sum(dim=1)  

        # negative scores
        batch_size, num_beams, seq_length = beams.size()
        negative_indices = torch.empty(batch_size, seq_length-1,dtype=torch.long, device=labels.device)

        for i in range(batch_size):
            positive_seq = labels[i, :5]  
            filtered_beams = [beam[1:] for beam in beams[i] if not torch.equal(beam[1:], positive_seq)]
            negative_seq = random.choice(filtered_beams) 
            negative_indices[i] = negative_seq

        #print(negative_indices)
        negative_logits = model(inputs.get("input_ids"), labels=negative_indices).logits 
        negative_probs = F.log_softmax(negative_logits, dim=-1)
        negative_selected_probs = torch.gather(negative_probs, 2, negative_indices.unsqueeze(-1)).squeeze(-1)
        negative_scores = negative_selected_probs.sum(dim=1)  

        losses = F.relu(negative_scores - positive_scores + margin)
        loss = losses.mean()

        return loss


    def prefix_ltr_loss(self, model, inputs, logits, labels, temperature, margin):

        if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
            model = model.module

        prefix_loss = []
        for prefix_length in range(1, 5):
            num_beams = 10
            generation_config = GenerationConfig(
                num_beams=num_beams,
                max_new_tokens=prefix_length,
                num_return_sequences=num_beams,
                early_stopping=True,
                use_cache=False,
            )
            beams = model.generate(inputs.get("input_ids"),
                                    generation_config=generation_config,
                                    prefix_allowed_tokens_fn=self.train_allowed_tokens
                                    ).reshape(inputs.get("input_ids").shape[0], 10, -1)
            
            # positive scores
            positive_logits = logits[:,:prefix_length,:]
            positive_probs = F.log_softmax(positive_logits, dim=-1)
            positive_selected_probs = torch.gather(positive_probs, 2, labels[:,:prefix_length].unsqueeze(-1)).squeeze(-1)
            positive_scores = positive_selected_probs.sum(dim=1)  

            # print(prefix_length)
            # print(labels[0,:prefix_length])
            # print(positive_scores[0])
            # print(beams[0])
            # negative scores
            batch_size, num_beams, seq_length = beams.size()
            negative_indices = torch.empty(batch_size, seq_length-1, dtype=torch.long, device=labels.device)

            for i in range(batch_size):
                positive_seq = labels[i, :prefix_length]  
                filtered_beams = [beam[1:] for beam in beams[i] if not torch.equal(beam[1:], positive_seq)]
                negative_seq = random.choice(filtered_beams) 
                negative_indices[i] = negative_seq

            #print(negative_indices[0])
            negative_logits = model(inputs.get("input_ids"), labels=negative_indices).logits / temperature
            negative_probs = F.log_softmax(negative_logits, dim=-1)
            negative_selected_probs = torch.gather(negative_probs, 2, negative_indices.unsqueeze(-1)).squeeze(-1)
            negative_scores = negative_selected_probs.sum(dim=1)  
            #print(negative_scores[0])

            losses = F.relu(negative_scores - positive_scores + margin)
            loss = losses.mean()
            prefix_loss.append(loss)
        
        return sum(prefix_loss) / len(prefix_loss)


    def multi_ltr_loss(self, model, inputs, logits, labels):
        
        if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
            model = model.module

        num_beams = 10
        generation_config = GenerationConfig(
            num_beams=num_beams,
            max_new_tokens=10,
            num_return_sequences=num_beams,
            early_stopping=True,
            use_cache=False,
        )
        beams = model.generate(inputs.get("input_ids"),
                                generation_config=generation_config,
                                prefix_allowed_tokens_fn=self.train_allowed_tokens
                                ).reshape(inputs.get("input_ids").shape[0], num_beams, -1)
        
        # positive scores
        positive_logits = logits[:,:5,:]
        positive_probs = F.log_softmax(positive_logits, dim=-1)
        positive_selected_probs = torch.gather(positive_probs, 2, labels[:,:5].unsqueeze(-1)).squeeze(-1)
        positive_scores = positive_selected_probs.sum(dim=1)  

        # negative scores
        batch_size, num_beams, seq_length = beams.size()

        losses = []
        for i in range(batch_size):
            positive_seq = labels[i, :5]
            filtered_beams = [beam[1:] for beam in beams[i] if not torch.equal(beam[1:], positive_seq)]
            input_i = inputs.get("input_ids")[i].unsqueeze(0).repeat(len(filtered_beams), 1)
            filtered_beams = torch.stack(filtered_beams, dim=0)
            #print(filtered_beams)
            negative_logits = model(input_i, labels=filtered_beams).logits
            negative_probs = F.log_softmax(negative_logits, dim=-1)
            negative_selected_probs = torch.gather(negative_probs, 2, filtered_beams.unsqueeze(-1)).squeeze(-1) 
            negative_scores = negative_selected_probs.sum(dim=1)
            #print(negative_scores)
            total_scores = torch.cat((positive_scores[i].unsqueeze(0), negative_scores), dim=0)
            total_prob = F.softmax(total_scores, dim=0)
            target = torch.zeros_like(total_scores)
            target[0] = 1
            losses.append(F.cross_entropy(total_prob, target))

        loss = torch.stack(losses).mean()

        return loss


