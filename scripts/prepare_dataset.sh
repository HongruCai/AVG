python -u prepare_dataset.py \
    --code_file RQ-VAE/output/rqvae_mscoco/t5large_VT_1024-512_1-c1024_e5000_lr0.0001_mse/mscoco_codes.json \
    --dataset mscoco \
    --output_dir mscoco_t5large \
    --pseudo_file data/mscoco/pseudo_query.json \