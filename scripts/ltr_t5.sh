deepspeed --master_port=29511 train_t5_ltr.py \
    --data_path data/flickr/flickr_codes_1024 \
    --output_dir output/flickr_ltr \
    --model_name_path  output/flickr/t5-base/t5-base_flickr_codes_1024/checkpoint-11720 \
    --train_epoch 5 \
    --learning_rate 1e-4 \
    --train_batch_size 32 \
    --wandb_log_freq 1 \
    --source_length 128 \
    --target_length 8 \
    --gen_len 20 \
    --warmup_ratio 0.1 \
    --eval_strategy epoch \
    --save_strategy no \
    --save_total_limit 1 \
    --logging_steps 100 \
    --deepseed_config config/t5_ds_config.json \
    --gradient_accumulation_steps 4 \
    --temperature 1.0 \
    --ltr_loss_factor 0.5 \
    --margin 5.0 \