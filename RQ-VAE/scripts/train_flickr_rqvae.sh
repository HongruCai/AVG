python -u train.py \
  --version test\
  --epochs 5000 \
  --dropout_prob 0.25 \
  --num_emb_list 1024 \
  --e_dim 768 \
  --layers 1024 512 \
  --device cuda:0 \
  --dataset flickr \
  --ckpt_dir ./output/rqvae_flickr \
  --code_length 4 \
  --bn \
  --kmeans_init \
  --use_cap \
  --use_pseudo \


