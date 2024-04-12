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
from datasets import load_dataset
from peft import TaskType, LoraConfig, get_peft_model, PeftModel
import argparse

from utils import LLaMaDataset, TrainerwithTemperature, LlaMaTrainerwithTemperature


def parse_args():
    parser = argparse.ArgumentParser(description="Train T5 model with RQ-VAE codes")

    parser.add_argument('--data_path', type=str, default='data/flickr/flickr_codes', help='data path')
    parser.add_argument('--output_dir', type=str, default='output/flickr', help='output directory')
    parser.add_argument('--model_name', type=str, default='t5-base', help='model name')
    parser.add_argument('--train_epoch', type=int, default=100, help='number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--train_batch_size', type=int, default=128, help='training batch size')
    parser.add_argument('--code_book_size', type=int, default=1024, help='code book size')
    parser.add_argument('--code_book_num', type=int, default=1, help='number of code books')
    parser.add_argument('--add_embedding', action='store_true', help='add rq_embedding to tokenizer')
    parser.add_argument('--wandb_log_freq', type=int, default=5, help='wandb log frequency')
    parser.add_argument('--source_length', type=int, default=128, help='source length')
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='warmup ratio')
    parser.add_argument('--eval_strategy', type=str, default='epoch', help='evaluation strategy')
    parser.add_argument('--save_strategy', type=str, default='epoch', help='save strategy')
    parser.add_argument('--save_total_limit', type=int, default=5, help='save total limit')
    parser.add_argument('--logging_steps', type=int, default=100, help='logging steps')
    parser.add_argument('--deepseed_config', type=str, default=None, help='deepspeed config file')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='gradient accumulation steps')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank')
    parser.add_argument('--temperature', type=float, default=1.0, help='softmax temperature')
    parser.add_argument('--float16', action='store_true', help='use float16')
    parser.add_argument('--bf16', action='store_true', help='use bf16')
    return parser.parse_args()


if __name__ == '__main__':

    train_args = parse_args()
    data_path = train_args.data_path

    print('training on: ', data_path)

    model_name = train_args.model_name
    train_source_file = data_path + '/train.source'
    train_target_file = data_path + '/train.target'
    val_source_file = data_path + '/val.source'
    val_target_file = data_path + '/val.target'
    test_source_file = data_path + '/test.source'
    test_target_file = data_path + '/test.target'
    
    train_epoch = train_args.train_epoch
    learning_rate = train_args.learning_rate
    train_batch_size = train_args.train_batch_size
    code_book_size = train_args.code_book_size
    code_book_num = train_args.code_book_num
    add_embedding = train_args.add_embedding
    wandb_log_freq = train_args.wandb_log_freq
    source_length = train_args.source_length
    output_dir = str(data_path.split('/')[-1])+'_c'+str(code_book_size)+'_ep'+str(train_epoch)+'_lr'+str(learning_rate)+'_bch'+str(train_batch_size)

    output_dir_name = train_args.output_dir + '/' + train_args.model_name.split('/')[-1] + '/' + output_dir
    
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    local_rank = int(os.environ.get("LOCAL_RANK") or 0)

    if local_rank == 0:
            wandb.login()
            wandb.init(project='llama',name=output_dir)
    if ddp:
        device_map = {"": local_rank}

    tokenizer = LlamaTokenizer.from_pretrained(
        model_name,
        #model_max_length=max_length, 
        padding_side="left"
        )
    tokenizer.padding_side = "left" 
    tokenizer.pad_token_id = 0

    if train_args.float16:
        torch_dtype = torch.float16
    elif train_args.bf16:
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32
    
    config = LlamaConfig.from_pretrained(model_name)
    model = LlamaForCausalLM.from_pretrained(model_name, 
                                             torch_dtype=torch_dtype,
                                             config=config,
                                             device_map=device_map)

    model.config.use_cache = False
    model.config.pretraining_tp = 1


    extra_tokens = []
    for count in range(code_book_size):
        extra_tokens.append('c_'+str(count))
    tokenizer.add_tokens(extra_tokens)
    model.resize_token_embeddings(len(tokenizer))

    if add_embedding:
        rq_emb = torch.load(data_path+'/codebook_embedding.pt')
        token_embeddings = model.get_input_embeddings()
        assert rq_emb.size(0) == code_book_size * code_book_num
        original_vocab_size = len(tokenizer) - code_book_size * code_book_num
        token_embeddings.weight.data[original_vocab_size:original_vocab_size + code_book_size * code_book_num] = rq_emb
        print('rq_embedding added')

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        modules_to_save=['embed_tokens','lm_head'],
        lora_dropout=0.1,
        bias="none",
        inference_mode=False,
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()


    reporter =  ['wandb'] if local_rank == 0 else "none"
    #reporter =  'none'
    training_args = TrainingArguments(
        output_dir=output_dir_name,

        num_train_epochs=train_epoch,         
        per_device_train_batch_size=train_batch_size, 
        per_device_eval_batch_size=train_batch_size, 
        dataloader_num_workers=10,
                
        # adafactor=True,
        # optim='adafactor',
        warmup_ratio=train_args.warmup_ratio,
        learning_rate=learning_rate,
        # weight_decay=0.01,   
                   
        logging_dir=output_dir_name+'/logs/',
        report_to=reporter,
        evaluation_strategy=train_args.eval_strategy,
        #eval_steps=1000,
        
        save_strategy=train_args.save_strategy,
        #save_steps=1000,
        save_total_limit=train_args.save_total_limit,

        logging_steps=train_args.logging_steps,

        deepspeed=train_args.deepseed_config,
        gradient_accumulation_steps=train_args.gradient_accumulation_steps,
        fp16=train_args.float16,
        bf16=train_args.bf16,
    )
    model.config.use_cache = False
 
    train_dataset = LLaMaDataset(tokenizer, train_source_file, train_target_file)
    val_dataset = LLaMaDataset(tokenizer, val_source_file, val_target_file)
    test_dataset = LLaMaDataset(tokenizer, test_source_file, test_target_file)

    #print(train_dataset[0])
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding='max_length', max_length=source_length, return_tensors="pt")

    if local_rank == 0:
        os.makedirs(output_dir_name, exist_ok=True)
        logging.basicConfig(filename=output_dir_name+'/training_log.log', level=logging.INFO, format='%(asctime)s - %(message)s')
        logger = logging.getLogger(__name__)

        logger.info('traing arguments: '+str(train_args))
        logger.info('training dataset size: '+str(len(train_dataset)))
        logger.info('validation dataset size: '+str(len(val_dataset)))
        logger.info('test dataset size: '+str(len(test_dataset)))
        logger.info('transfomers training_args: '+str(training_args))

    trainer = LlaMaTrainerwithTemperature(
        temperature=train_args.temperature,
        vocab_size=len(tokenizer),

        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

