python -u prepare_emb.py \
  --root_dir data/flickr/images \
  --caption_file data/flickr/flickr_split_captions.json \
  --pseudo_caption_file data/flickr/pseudo_query.json \
  --batch_size 1024 \
  --clip_model openai/clip-vit-large-patch14-336 \
  --save_path data/flickr/emb


