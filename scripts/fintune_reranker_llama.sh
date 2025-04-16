# pretrained model name: AILab-CVC/seed-llama-8b-sft
# You can download first or directly use the pretrained model from Hugging Face

deepspeed --master_port=23333 finetune_reranker_llama.py \
    --data_path data/flickr/flickr \
    --output_dir output/reranker/flickr \
    --model_name models/pretrained/seed_llama_8b_sft \
    --train_epoch 3 \
    --learning_rate 3e-5 \
    --train_batch_size 1 \
    --wandb_log_freq 100 \
    --source_length 64 \
    --warmup_ratio 0.1 \
    --eval_strategy steps \
    --save_strategy steps \
    --save_total_limit 3 \
    --logging_steps 100 \
    --deepseed_config config/llama_ds_config.json \
    --gradient_accumulation_steps 16 \
    --bf16 \
    --top_k 100 \
    --group_size 8 \
    --train_predictions data/flickr/train_predictions.json \
    --test_predictions data/flickr/test_predictions.json \
    --id_file data/flickr/avg_to_seed.json \