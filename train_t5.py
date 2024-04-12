import os
from torch.utils.data import Dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer, T5Config
from transformers import Trainer, TrainingArguments, TrainerCallback, DataCollatorWithPadding
import torch
import logging
import wandb
import argparse
from utils import QueryEvalCallback, TrainerwithTemperature, T5Dataset
from peft import TaskType, LoraConfig, get_peft_model, PeftModel


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
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--log_freq', type=int, default=5, help='eval log frequency')
    parser.add_argument('--source_length', type=int, default=128, help='source length')
    parser.add_argument('--target_length', type=int, default=8, help='target length')
    parser.add_argument('--gen_len', type=int, default=20, help='generation length')
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='warmup ratio')
    parser.add_argument('--eval_strategy', type=str, default='epoch', help='evaluation strategy')
    parser.add_argument('--save_strategy', type=str, default='epoch', help='save strategy')
    parser.add_argument('--save_total_limit', type=int, default=5, help='save total limit')
    parser.add_argument('--logging_steps', type=int, default=100, help='logging steps')
    parser.add_argument('--deepseed_config', type=str, default=None, help='deepspeed config file')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='gradient accumulation steps')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank')
    parser.add_argument('--temperature', type=float, default=1.0, help='softmax temperature')
    parser.add_argument('--add_prefix', action='store_true', help='add prefix to inputs')
    parser.add_argument('--float16', action='store_true', help='use float16')
    parser.add_argument('--bf16', action='store_true', help='use bf16')
    parser.add_argument('--lora', action='store_true', help='use lora')
    
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
    dropout_rate = train_args.dropout_rate
    log_freq = train_args.log_freq
    source_length = train_args.source_length
    target_length = train_args.target_length
    gen_len = train_args.gen_len

    rq_signal = 'embadded' if add_embedding else 'noemb'
    output_dir = model_name.split('/')[-1]+'_'+str(data_path.split('/')[-1])+'_c'+str(code_book_size)+'_ep'+str(train_epoch)+'_lr'+str(learning_rate)+'_bch'+str(train_batch_size)+'_'+ rq_signal

    local_rank = int(os.environ.get("LOCAL_RANK") or 0)

    if local_rank == 0:
        wandb.login()
        wandb.init(project='huggingface',name=output_dir)
    
    output_dir_name = train_args.output_dir + '/' + train_args.model_name.split('/')[-1] + '/' + output_dir

    tokenizer = T5Tokenizer.from_pretrained(model_name)
    config = T5Config.from_pretrained(model_name)
    config.dropout_rate = dropout_rate   
    if train_args.float16:
        torch_dtype = torch.float16
    elif train_args.bf16:
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32
    model = T5ForConditionalGeneration.from_pretrained(model_name, 
                                                       torch_dtype=torch_dtype,
                                                       config=config)

    prefix = ['a_','b_','c_','d_']

    extra_tokens = []
    if code_book_num == 1:
        for count in range(code_book_size):
            extra_tokens.append('c_'+str(count))
    else:
        for code_book in range(code_book_num):
            for count in range(code_book_size):
                extra_tokens.append(prefix[code_book]+str(count))
    print('number of extra tokens: ', len(extra_tokens))
    tokenizer.add_tokens(extra_tokens)
    model.resize_token_embeddings(len(tokenizer))

    if add_embedding:
        rq_emb = torch.load(data_path+'/codebook_embedding.pt')
        token_embeddings = model.get_input_embeddings()
        assert rq_emb.size(0) == code_book_size * code_book_num
        original_vocab_size = len(tokenizer) - code_book_size * code_book_num
        token_embeddings.weight.data[original_vocab_size:original_vocab_size + code_book_size * code_book_num] = rq_emb
        print('codebook_embedding added')


    if train_args.lora:
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
    #reporter =  ['tensorboard'] if local_rank == 0 else "none"
    training_args = TrainingArguments(
        output_dir=output_dir_name,

        num_train_epochs=train_epoch,         
        per_device_train_batch_size=train_batch_size, 
        per_device_eval_batch_size=train_batch_size, 
        dataloader_num_workers=10,
                
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
        #load_best_model_at_end=True,

        logging_steps=train_args.logging_steps,

        deepspeed=train_args.deepseed_config,
        gradient_accumulation_steps=train_args.gradient_accumulation_steps,
        
        fp16=train_args.float16,
        bf16=train_args.bf16,
    )
    model.config.use_cache = False

    train_dataset = T5Dataset(tokenizer, train_source_file, train_target_file, max_source_len=source_length, max_target_len=target_length, 
                              add_prefix=train_args.add_prefix)
    val_dataset = T5Dataset(tokenizer, val_source_file, val_target_file, max_source_len=source_length, max_target_len=target_length, 
                            add_prefix=train_args.add_prefix)
    test_dataset = T5Dataset(tokenizer, test_source_file, test_target_file, max_source_len=source_length, max_target_len=target_length, 
                             add_prefix=train_args.add_prefix)
    sub_train_dataset = T5Dataset(tokenizer, train_source_file, train_target_file, max_source_len=source_length, max_target_len=target_length, 
                                  add_prefix=train_args.add_prefix,subset_size=1000)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding='max_length', max_length=source_length)

    os.makedirs(output_dir_name, exist_ok=True)
    logging.basicConfig(filename=output_dir_name+'/training_log.log', level=logging.INFO, format='%(asctime)s - %(message)s')
    logger = logging.getLogger(__name__)

    if local_rank == 0:
        logger.info('traing arguments: '+str(train_args))
        logger.info('training dataset size: '+str(len(train_dataset)))
        logger.info('validation dataset size: '+str(len(val_dataset)))
        logger.info('test dataset size: '+str(len(test_dataset)))
        logger.info('transfomers training_args: '+str(training_args))

    trainer = TrainerwithTemperature(
        temperature=train_args.temperature,

        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[QueryEvalCallback(local_rank=local_rank,
                                     test_dataset_1=sub_train_dataset, 
                                     test_dataset_2=test_dataset, 
                                     tgt_file=test_target_file,
                                     logger=logger, 
                                     batch_size=train_batch_size, 
                                     collator=data_collator, 
                                     tokenizer=tokenizer,
                                     wandb=wandb,
                                     log_freq=log_freq,
                                     gen_len=gen_len)],
    )

    trainer.train()

