python tools/test_reranker_llama.py \
    --device cuda \
    --base_model models/pretrained/seed_llama_8b_sft \
    --model_path output/reranker/flickr/seed_llama_8b_sft/xxx/checkpoint \
    --predictions_file data/flickr/test_predictions.json \
    --id_file data/flickr/avg_to_seed.json \
    --top_k 50 \
    --batch_size 4