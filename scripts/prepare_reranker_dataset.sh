accelerate launch tools/generate_predictions.py \
  --model_path output/retriever/flickr/t5-base/xx/xxx \
  --train_mode data/flickr/flickr_ft \
  --batch_size 256 \
  --num_beams 100 \
  --valid_modes train \
  --save_path data/flickr/train_predictions.json