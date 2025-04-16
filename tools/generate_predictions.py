from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    T5Config,
    DataCollatorWithPadding,
    GenerationConfig,
)
import torch
import os
import random
import numpy as np
from tqdm import tqdm
from typing import Dict, List
import json
import argparse
from accelerate import PartialState
from accelerate.utils import gather_object


class T5Dataset(Dataset):
    def __init__(
        self,
        tokenizer,
        source_file,
        target_file,
        max_source_len=128,
        max_target_len=8,
        subset_size=None,
    ):
        self.tokenizer = tokenizer
        self.source_texts = open(
            source_file, encoding="utf-8").read().splitlines()
        self.target_texts = open(
            target_file, encoding="utf-8").read().splitlines()
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len
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

        source_encodings = self.tokenizer(
            source_text,
            padding="max_length",
            max_length=self.max_source_len,
            truncation=True,
            return_tensors="pt",
        )
        target_encodings = self.tokenizer(
            target_text,
            padding="max_length",
            max_length=self.max_target_len,
            truncation=True,
            return_tensors="pt",
        )
        # print(target_encodings)

        input_ids = source_encodings["input_ids"]
        labels = target_encodings["input_ids"]
        # labels[labels == self.tokenizer.pad_token_id] = -100

        return {"input_ids": input_ids.squeeze(), "labels": labels.squeeze()}


def load_data(source_file, target_file, sub_size=None):
    target_lines = open(target_file, encoding="utf-8").read().splitlines()
    source_lines = open(source_file, encoding="utf-8").read().splitlines()

    if sub_size is not None and sub_size < len(source_lines):
        indeces = np.random.choice(len(source_lines), sub_size, replace=False)
        source_lines = [source_lines[i] for i in indeces]
        target_lines = [target_lines[i] for i in indeces]

    source = []
    target = []
    for i in range(len(source_lines)):
        source_text = source_lines[i]
        target_text = target_lines[i]
        source.append(source_text)
        target.append(target_text)

    return source, target


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

    @ staticmethod
    def load_from_dict(trie_dict):
        trie = Trie()
        trie.trie_dict = trie_dict
        trie.len = sum(1 for _ in trie)
        return trie

    @ staticmethod
    def _add_to_trie(sequence: List[int], trie_dict: Dict):
        if sequence:
            if sequence[0] not in trie_dict:
                trie_dict[sequence[0]] = {}
            Trie._add_to_trie(sequence[1:], trie_dict[sequence[0]])

    @ staticmethod
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


def prefix_allowed_tokens_fn(candidate_trie):
    def prefix_allowed_tokens(batch_id, sentence):
        sentence = sentence.tolist()
        # print(sentence)
        trie_out = candidate_trie.get(sentence)
        return trie_out

    return prefix_allowed_tokens


def load_codes(target_file):
    with open(target_file, "r", encoding="utf-8") as tgt_file:
        target_lines = tgt_file.readlines()
    res = []
    for i in range(len(target_lines)):
        if i % 5 == 0 and target_lines[i].strip() not in res:
            res.append(target_lines[i].strip())
    target_lines = res

    return target_lines


def load_queries(source_file):
    queries = open(source_file, encoding="utf-8").read().splitlines()
    return queries


def main(args):

    distributed_state = PartialState()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    tokenizer = T5Tokenizer.from_pretrained(args.model_path)
    model = T5ForConditionalGeneration.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, device_map=distributed_state.device)

    print("tokenizer loaded from " + args.model_path)
    print("model loaded from " + args.model_path)

    # model.to(device)
    model.eval()

    valid_modes = args.valid_modes.split(",")  
    for valid_mode in valid_modes:
        source_file = f"{args.train_mode}/{valid_mode}.source"
        tgt_file = f"{args.train_mode}/{valid_mode}.target"

        print("Loading data from", source_file, "and", tgt_file)
        dataset = T5Dataset(
            tokenizer,
            source_file,
            tgt_file,
            subset_size=args.sample_num,
            max_source_len=128,
            max_target_len=8
        )

        data_collator = DataCollatorWithPadding(
            tokenizer=tokenizer, padding="max_length", max_length=128
        )

        dataloader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True, collate_fn=data_collator
        )

        code_list = load_codes(tgt_file)
        condidate_trie = Trie([[0] + tokenizer.encode(x) for x in code_list])
        print("Trie loaded, possible response count:", len(condidate_trie))
        prefix_allowed_tokens = prefix_allowed_tokens_fn(condidate_trie)

        recall_count_at_1 = 0
        recall_count_at_5 = 0
        recall_count_at_10 = 0
        results = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"[Evaluating {valid_mode} queries"):
                inputs = batch

                generation_config = GenerationConfig(
                    num_beams=args.num_beams,
                    max_new_tokens=8,
                    num_return_sequences=args.num_beams,
                    early_stopping=True,
                    use_cache=True,
                )

                with distributed_state.split_between_processes(list(zip(
                    inputs["input_ids"], inputs["labels"]
                ))) as local_batches:
                    local_results = []

                    for input_ids, label in local_batches:
                        input_ids = input_ids.unsqueeze(0).to(distributed_state.device)
                        label = label.to(distributed_state.device)

                        beams = model.generate(
                            input_ids,
                            generation_config=generation_config,
                            prefix_allowed_tokens_fn=prefix_allowed_tokens,
                        ).reshape(1, args.num_beams, -1)

                        rank_list = tokenizer.batch_decode(beams[0], skip_special_tokens=True)
                        rank_list = [x.split(" ") for x in rank_list]

                        label[label == -100] = tokenizer.pad_token_id
                        label = tokenizer.decode(label, skip_special_tokens=True).strip().split(" ")
                        input_text = tokenizer.decode(input_ids[0], skip_special_tokens=True).strip()

                        result_entry = {
                            "input": input_text,
                            "label": label,
                            "predictions": rank_list,
                        }
                        local_results.append(result_entry)

                local_results = gather_object(local_results)

                for result_entry in local_results:
                    results.append(result_entry)
                    # if distributed_state.is_main_process and len(results) % 100 == 0:
                    #     with open(args.save_path, "w", encoding="utf-8") as f:
                    #         json.dump(results, f)
                    label = result_entry["label"]
                    rank_list = result_entry["predictions"]

                    hits = [i for i, x in enumerate(rank_list) if x == label and i < 10]
                    if hits:
                        recall_count_at_10 += 1
                        if hits[0] < 5:
                            recall_count_at_5 += 1
                        if hits[0] == 0:
                            recall_count_at_1 += 1

        if distributed_state.is_main_process:
            with open(args.save_path, "w", encoding="utf-8") as f:
                json.dump(results, f)
            total = len(results)
            print(f"Recall@1: {recall_count_at_1 / total:.4f}")
            print(f"Recall@5: {recall_count_at_5 / total:.4f}")
            print(f"Recall@10: {recall_count_at_10 / total:.4f}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--train_mode", type=str, required=True)
    parser.add_argument("--sample_num", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_beams", type=int, default=50)
    parser.add_argument("--valid_modes", type=str, default="test")
    parser.add_argument("--save_path", type=str, default="results.json")

    args = parser.parse_args()
    main(args)
