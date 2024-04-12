python -u train.py \
  --version t5large \
  --epochs 5000 \
  --dropout_prob 0.25 \
  --num_emb_list 1024 \
  --e_dim 1024 \
  --layers 1024 512 \
  --device cuda:0 \
  --dataset mscoco \
  --ckpt_dir ./output/rqvae_mscoco/ \
  --bn \
  --kmeans_init \
  --use_cap \
  --use_pseudo


