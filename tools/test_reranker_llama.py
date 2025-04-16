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
from accelerate import PartialState
from accelerate.utils import gather_object
from collections import defaultdict

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


def compute_similarity(model, tokenizer, input_text, predictions, device, id_file):
    id_file = json.load(open(id_file, 'r'))

    predictions = [id_file[pred] for pred in predictions]

    prefix = format_input(input_text, predictions[0], predictions[1:])

    similarities = []
    inputs = tokenizer(prefix, return_tensors="pt",  # padding="max_length",
                       max_length=64, truncation=True)
    input_ids = inputs.input_ids.to(device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        similarities = outputs.float().cpu().numpy().tolist()

    return similarities


def rerank_predictions(results, model, tokenizer, device, sample, top_k=10, id_file='data/avg_to_seed.json'):
    reranked_results = []
    if sample is not None:
        results = random.sample(results, sample)
    for result in tqdm(results, desc="Reranking predictions"):
        input_text = result['input']
        predictions = result['predictions'][:top_k]

        similarities = compute_similarity(
            model, tokenizer, input_text, predictions, device, id_file)

        sorted_predictions = [x for _, x in sorted(
            zip(similarities, predictions), reverse=True)]

        reranked_result = {
            "input": result['input'],
            "label": result['label'],
            "predictions": sorted_predictions,
            "original_predictions": result['predictions'][:top_k],
        }
        reranked_results.append(reranked_result)

    return reranked_results


def batch_compute_similarity(model, tokenizer, inputs, predictions, device, id_file):

    predictions = [[id_file[pred] for pred in preds] for preds in predictions]

    batch_prefix = [format_input(input_text, preds[0], preds[1:]) for input_text, preds in zip(inputs, predictions)]

    batch_prefix = [item for sublist in batch_prefix for item in sublist]  # flatten

    input_ids = tokenizer(batch_prefix, return_tensors="pt", padding="max_length", truncation=True, max_length=64).input_ids.to(device)

    similarities = []
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        similarities = outputs.float().cpu().numpy().tolist()

    similarities = [similarities[i:i+len(predictions[0])] for i in range(0, len(similarities), len(predictions[0]))]

    return similarities


def batch_rerank_predictions(results, model, tokenizer, bacth_size, device, sample, top_k=10, id_file='data/avg_to_seed.json'):
    reranked_results = []
    if sample is not None:
        results = random.sample(results, sample)
    id_file = json.load(open(id_file, 'r'))

    batch_size = bacth_size  
    for i in tqdm(range(0, len(results), batch_size)):
        batch_results = results[i:i+batch_size]
        inputs = [res['input'] for res in batch_results]
        prediction_lists = [res['predictions'][:top_k] for res in batch_results]

        similarities = batch_compute_similarity(model, tokenizer, inputs, prediction_lists, device, id_file)

        for idx, (result, sim) in enumerate(zip(batch_results, similarities)):
            sorted_predictions = [x for _, x in sorted(zip(sim, prediction_lists[idx]), reverse=True)]

            reranked_result = {
                "input": result['input'],
                "label": result['label'],
                "predictions": sorted_predictions,
                "original_predictions": result['predictions'][:top_k],
            }
            reranked_results.append(reranked_result)

    return reranked_results


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



def parse_args():
    parser = argparse.ArgumentParser(
        description="Test reranker")

    parser.add_argument('--model_path', type=str,
                        default='output/flickr/seed_llama_8b_sft/ckpt', help='model path')
    parser.add_argument('--predictions_file', type=str,
                        default='data/flickr/flickr_codes', help='predictions path')
    parser.add_argument('--id_file', type=str,
                        default='data/flickr/avg_to_seed.json', help='id file')
    # multi-gpu or batch inference is not supported
    parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--base_model', type=str,
                        default='pretrained/seed_llama_8b_sft', help='base model')
    parser.add_argument('--sample_num', type=int,
                        default=None, help='number of samples')
    parser.add_argument('--num_beams', type=int,
                        default=10, help='number of beams')
    parser.add_argument('--split', type=str, default='test',
                        help='split to evaluate')
    parser.add_argument('--top_k', type=int,
                        default=10, help='rerank from top k predictions')
    parser.add_argument('--batch_size', type=int,
                        default=32, help='batch size')
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()
    device = args.device

    base_model = args.base_model
    model_path = args.model_path

    # distributed_state = PartialState()


    tokenizer = LlamaTokenizer.from_pretrained(
        model_path,
        # model_max_length=max_length,
    )
    tokenizer.pad_token = tokenizer.unk_token

    model = LlamaWithMLP.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        # device_map=distributed_state.device
        # device_map=device_map,
    )

    model = PeftModel.from_pretrained(
        model,
        model_path,
        torch_dtype=torch.float16,
        # device_map=distributed_state.device
        # device_map=device_map,
    )

    model.to(device)
    model.eval()

    print('tokenizer loaded from '+model_path)
    print('model loaded from '+model_path)

    with open(args.predictions_file, 'r') as f:
        results = json.load(f)
    for result in results:
        result['predictions'] = [' '.join(pred) for pred in result['predictions']]
        result['label'] = ' '.join(result['label'])


    top_k = args.top_k
    print("Reranking from top", top_k, "predictions")

    reranked_results = batch_rerank_predictions(
        results, model, tokenizer, args.batch_size, device, sample=None, top_k=top_k, id_file=args.id_file)

    # Evaluate recall
    recall_count_at_1 = 0
    recall_count_at_5 = 0
    recall_count_at_10 = 0

    for result in reranked_results:
        label = result['label']
        sorted_predictions = result['predictions']


        hits = [i for i, x in enumerate(sorted_predictions) if x == label]
        hits = [x for x in hits if x < 10]

        if len(hits) != 0:
            recall_count_at_10 += 1
            if hits[0] < 5:
                recall_count_at_5 += 1
            if hits[0] == 0:
                recall_count_at_1 += 1

    total_samples = len(reranked_results)
    hits_at_1_data = recall_count_at_1 / total_samples
    hits_at_5_data = recall_count_at_5 / total_samples
    hits_at_10_data = recall_count_at_10 / total_samples

    print("Reranked results:")
    print("Recall@1: ", hits_at_1_data)
    print("Recall@5: ", hits_at_5_data)
    print("Recall@10: ", hits_at_10_data)
