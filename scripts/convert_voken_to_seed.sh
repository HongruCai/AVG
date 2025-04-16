
python tools/convert_seed.py \
  --image_caption_file RQ-VAE/output/rqvae_flickr/xxx/flickr_codes.json \
  --output_file data/flickr/avg_to_seed.json \
  --image_dirs RQ-VAE/data/flickr/flickr30k/Images \
  --tokenizer_cfg config/seed_llama_tokenizer_hf.yaml \
  --transform_cfg config/clip_transform.yaml \
  --device cuda
