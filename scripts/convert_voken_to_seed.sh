
python tools/convert_seed.py \
  --image_caption_file RQ-VAE/output/rqvae_flickr/250412final_test_VT_1024-512_1-c1024_e1500_lr0.0001_mse/flickr_codes.json \
  --output_file data/flickr/avg_to_seed_ft.json \
  --image_dirs RQ-VAE/data/flickr/flickr30k/Images \
  --tokenizer_cfg config/seed_llama_tokenizer_hf.yaml \
  --transform_cfg config/clip_transform.yaml \
  --device cuda
