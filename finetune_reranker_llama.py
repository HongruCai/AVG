from datetime import datetime
import sys
import os
from torch.utils.data import Dataset
import transformers
from transformers import T5ForConditionalGeneration, T5Tokenizer, T5Config
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig
from transformers import Trainer, TrainingArguments, TrainerCallback, DataCollatorWithPadding, GenerationConfig
from transformers import (
    BitsAndBytesConfig,
    HfArgumentParser,
    pipeline,
    logging,
)
import torch
import logging
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard.writer import SummaryWriter
import random
import wandb
# import deepspeed
from typing import Dict, List
from peft import TaskType, LoraConfig, get_peft_model, PeftModel
import argparse
import json
import torch.nn as nn
import hydra

from models.seed_llama_tokenizer import SeedLlamaTokenizerOnly
import pyrootutils
import os
import torch

from omegaconf import OmegaConf
import json
from typing import Optional
import transformers
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig
from datetime import datetime

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

BOI_TOKEN = '<img>'
EOI_TOKEN = '</img>'
IMG_TOKEN = '<img_{:05d}>'



def format_input(input_text, label, predictions):
    label_token = [IMG_TOKEN.format(item) for item in label]

    pred_tokens = [[IMG_TOKEN.format(item)
                    for item in pred] for pred in predictions]

    img_tokens = [label_token] + pred_tokens

    prefix = [
        f'USER: Image: {BOI_TOKEN}{"".join(tokens)}{EOI_TOKEN} Query: {input_text} ASSISTANT:' for tokens in img_tokens]

    return prefix


class LlamaRankingDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        source_file,
        id_file,
        max_source_len=128,
        interval=8,
        sample_from='pred',
        top_k=100
    ):
        self.tokenizer = tokenizer
        self.source_file = json.load(open(source_file, 'r'))
        self.id_file = json.load(open(id_file, 'r'))
        self.max_source_len = max_source_len
        self.interval = interval
        self.sample_from = sample_from
        self.all_vokens_file = list(self.id_file.keys())
        self.top_k = top_k

    def __len__(self):
        return len(self.source_file)

    def __getitem__(self, idx):
        query_text = self.source_file[idx]['input']
        label_text = self.source_file[idx]['label']
        label_text = ' '.join(label_text)
        seed_label_text = self.id_file[label_text]

        if self.sample_from == 'pred':
            predictions = self.source_file[idx]['predictions'][:self.top_k]
            predictions = [' '.join(pred) for pred in predictions]
            predictions = [
                pred for pred in predictions if pred in self.all_vokens_file]
        else:
            predictions = self.all_vokens_file

        false_predictions = [
            pred for pred in predictions if pred != label_text]

        false_predictions = random.sample(
            false_predictions, self.interval-1)
        false_predictions = [self.id_file[pred]
                             for pred in false_predictions]

        group_texts = format_input(
            query_text, seed_label_text, false_predictions)

        # print(group_texts)
        assert len(group_texts) == self.interval

        texts = [self.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_source_len,
            truncation=True,
            return_tensors="pt",
        ) for text in group_texts]

        input_ids = torch.cat([enc["input_ids"] for enc in texts])

        return {
            "input_ids": input_ids,
        }


class LlamaWithMLP(LlamaForCausalLM):
    def __init__(self, config, mlp_hidden_dim=2048):
        super(LlamaWithMLP, self).__init__(config)
        self.final_mlp = nn.Sequential(
            nn.Linear(config.hidden_size, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, 1)  
        )

    def forward(self, input_ids, attention_mask=None, labels=None, *model_args, **kwargs):
        outputs = super(LlamaWithMLP, self).forward(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        last_hidden_state = hidden_states[:, -1, :]
        relevance_score = self.final_mlp(last_hidden_state).squeeze(-1)
        return relevance_score

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        model = super(LlamaWithMLP, cls).from_pretrained(
            pretrained_model_name_or_path, *model_args, **kwargs)
        if os.path.isfile(f"{pretrained_model_name_or_path}/mlp.pth"):
            mlp_state_dict = torch.load(
                f"{pretrained_model_name_or_path}/mlp.pth", map_location=torch.device('cpu'))
            model.final_mlp.load_state_dict(mlp_state_dict)
        else:
            for layer in model.final_mlp:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
        return model

    def save_pretrained(self, save_directory, **kwargs):
        super(LlamaWithMLP, self).save_pretrained(save_directory, **kwargs)
        torch.save(self.final_mlp.state_dict(), f"{save_directory}/mlp.pth")


class RankingTrainer(Trainer):
    def __init__(self, interval, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.interval = interval

    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids = inputs['input_ids']
        scores = model(input_ids=input_ids)
        interval = self.interval
        loss = []
        for i in range(len(scores) // interval):
            positive_scores = scores[i * interval]
            negative_scores = scores[i * interval + 1:i * interval + interval]

            exp_positive_scores = torch.exp(positive_scores)
            exp_negative_scores = torch.exp(negative_scores)

            sum_exp_scores = exp_positive_scores + \
                exp_negative_scores.sum(dim=0)
            loss.append(-torch.log(exp_positive_scores / sum_exp_scores))

        loss = torch.stack(loss).mean()
        return (loss, scores) if return_outputs else loss


def collate_fn(batch):
    # print(batch[0]["input_ids"].size())
    input_ids = torch.cat([item['input_ids'] for item in batch], dim=0)
    # print(input_ids.size())
    # assert input_ids[1] == batch[0]['input_ids'][1]
    # assert torch.equal(input_ids[5], batch[0]['input_ids'][5])
    return {
        'input_ids': input_ids,
        'labels': torch.zeros(input_ids.size(0)).to(input_ids.device)
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Seed-LLaMA for reranking")

    parser.add_argument('--data_path', type=str,
                        default='data/flickr/flickr_codes', help='data path')
    parser.add_argument('--output_dir', type=str,
                        default='output/flickr', help='output directory')
    parser.add_argument('--model_name', type=str,
                        default='t5-base', help='model name')
    parser.add_argument('--train_epoch', type=int,
                        default=100, help='number of training epochs')
    parser.add_argument('--learning_rate', type=float,
                        default=1e-3, help='learning rate')
    parser.add_argument('--train_batch_size', type=int,
                        default=1, help='training batch size')
    parser.add_argument('--wandb_log_freq', type=int,
                        default=5, help='wandb log frequency')
    parser.add_argument('--source_length', type=int,
                        default=128, help='source length')
    parser.add_argument('--warmup_ratio', type=float,
                        default=0.1, help='warmup ratio')
    parser.add_argument('--eval_strategy', type=str,
                        default='epoch', help='evaluation strategy')
    parser.add_argument('--save_strategy', type=str,
                        default='epoch', help='save strategy')
    parser.add_argument('--save_total_limit', type=int,
                        default=5, help='save total limit')
    parser.add_argument('--logging_steps', type=int,
                        default=100, help='logging steps')
    parser.add_argument('--deepseed_config', type=str,
                        default=None, help='deepspeed config file')
    parser.add_argument('--gradient_accumulation_steps',
                        type=int, default=1, help='gradient accumulation steps')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank')
    parser.add_argument('--temperature', type=float,
                        default=1.0, help='softmax temperature')
    parser.add_argument('--float16', action='store_true', help='use float16')
    parser.add_argument('--bf16', action='store_true', help='use bf16')
    parser.add_argument('--top_k', type=int, default=100, help='top k for negative sampling')
    parser.add_argument('--group_size', type=int, default=8, help='group size in one loss computation, inlucding the one positive and the rest negatives')
    parser.add_argument('--train_predictions', type=str, default='data/flickr/train_predictions.json', )
    parser.add_argument('--test_predictions', type=str, default='data/flickr/test_results.json', )
    parser.add_argument('--id_file', type=str, default='data/flickr/avg_to_seed.json', help='id file, convert the predictions ids to seed-llama ids')
                        
    return parser.parse_args()



if __name__ == '__main__':

    train_args = parse_args()
    data_path = train_args.data_path

    print('training on: ', data_path)

    model_name = train_args.model_name

    train_epoch = train_args.train_epoch
    learning_rate = train_args.learning_rate
    train_batch_size = train_args.train_batch_size
    wandb_log_freq = train_args.wandb_log_freq
    source_length = train_args.source_length
    current_time = datetime.now().strftime("%Y%m%d_%H%M")
    top_k = train_args.top_k
    output_dir = current_time+'_'+str(data_path.split('/')[-1])+'_top'+str(top_k)+'_ep'+str(
        train_epoch)+'_lr'+str(learning_rate)+'_bch'+str(train_batch_size)

    output_dir_name = train_args.output_dir + '/' + \
        train_args.model_name.split('/')[-1] + '/' + output_dir

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    local_rank = int(os.environ.get("LOCAL_RANK") or 0)

    if local_rank == 0:
        wandb.login()
        wandb.init(project='AVG_reranker', name=output_dir)


    # tokenizer_cfg_path = 'config/seed_llama_tokenizer_hf_only.yaml'
    # tokenizer_cfg = OmegaConf.load(tokenizer_cfg_path)
    # tokenizer = hydra.utils.instantiate(tokenizer_cfg, load_diffusion=False)
    # tokenizer.pad_token = tokenizer.unk_token

    tokenizer = SeedLlamaTokenizerOnly.from_pretrained('AILab-CVC/seed-tokenizer-2')
    tokenizer.fp16 = True
    tokenizer.load_diffusion = False
    tokenizer.encoder_url = 'https://huggingface.co/AILab-CVC/seed-tokenizer-2/resolve/main/seed_quantizer.pt'
    tokenizer.diffusion_path = 'stabilityai/stable-diffusion-2-1-unclip'

    print('seed llama tokenizer: initialized.')
    # tokenizer.padding_side = "left"


    if train_args.float16:
        torch_dtype = torch.float16
    elif train_args.bf16:
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32

    config = LlamaConfig.from_pretrained(model_name)
    model = LlamaWithMLP.from_pretrained(model_name,
                                         torch_dtype=torch_dtype,
                                         config=config,)

    model.config.use_cache = False

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        modules_to_save=['embed_tokens', 'lm_head', 'final_mlp',
                         'input_layernorm', 'post_attention_layernorm', 'norm'],
        target_modules=['q_proj', 'v_proj', 'k_proj',
                        'o_proj', 'gate_proj', 'down_proj', 'up_proj'],
        lora_dropout=0.05,
        bias="none",
        inference_mode=False,
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    reporter = ['wandb'] if local_rank == 0 else "none"
    # reporter = 'none'
    training_args = TrainingArguments(
        output_dir=output_dir_name,

        num_train_epochs=train_epoch,
        # max_steps=2000,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=train_batch_size,
        dataloader_num_workers=10,

        # adafactor=True,
        # optim='adafactor',
        lr_scheduler_type='cosine',
        warmup_ratio=train_args.warmup_ratio,
        learning_rate=learning_rate,
        # weight_decay=0.01,

        logging_dir=output_dir_name+'/logs/',
        report_to=reporter,
        evaluation_strategy=train_args.eval_strategy,
        eval_steps=400,

        save_strategy=train_args.save_strategy,
        save_steps=400,
        save_total_limit=train_args.save_total_limit,

        logging_steps=train_args.logging_steps,

        deepspeed=train_args.deepseed_config,
        gradient_accumulation_steps=train_args.gradient_accumulation_steps,
        fp16=train_args.float16,
        bf16=train_args.bf16,

        # load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        save_only_model=True,
    )
    model.config.use_cache = False


    train_dataset = LlamaRankingDataset(
        tokenizer=tokenizer,
        source_file=train_args.train_predictions,
        id_file=train_args.id_file,
        max_source_len=source_length,
        interval=train_args.group_size,
        sample_from='pred',
        top_k=top_k
    )
    test_dataset = LlamaRankingDataset(
        tokenizer=tokenizer,
        source_file=train_args.test_predictions,
        id_file=train_args.id_file,
        max_source_len=source_length,
        interval=train_args.group_size,
        sample_from='pred',
        top_k=top_k
    )

    if local_rank == 0:
        os.makedirs(output_dir_name, exist_ok=True)
        logging.basicConfig(filename=output_dir_name+'/training_log.log',
                            level=logging.INFO, format='%(asctime)s - %(message)s')
        logger = logging.getLogger(__name__)

        logger.info('traing arguments: '+str(train_args))
        logger.info('training dataset size: '+str(len(train_dataset)))
        logger.info('test dataset size: '+str(len(test_dataset)))
        logger.info('transfomers training_args: '+str(training_args))

    trainer = RankingTrainer(
        interval=train_args.group_size,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=collate_fn,
    )

    trainer.train()
