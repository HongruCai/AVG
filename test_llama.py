from transformers import GenerationConfig
from transformers import LlamaTokenizer, LlamaForCausalLM
import torch
import os
from tqdm import tqdm
import argparse
from peft import PeftModel
from utils import Trie, llama_prefix_allowed_tokens_fn, load_response, load_prompt


def parse_args():
    parser = argparse.ArgumentParser(description="Test Llama model with RQ-VAE codes")

    parser.add_argument('--model_path', type=str, default='output/flickr/llama-2/', help='model path')
    parser.add_argument('--data_path', type=str, default='data/flickr/flickr_codes', help='data path')
    parser.add_argument('--device', type=str, default='cuda', help='device') # multi-gpu or batch inference is not supported
    parser.add_argument('--base_model', type=str, default='meta-llama/Llama-2-7b-hf', help='base model')
    parser.add_argument('--sample_num', type=int, default=None, help='number of samples')
    parser.add_argument('--num_beams', type=int, default=10, help='number of beams')
    parser.add_argument('--split', type=str, default='test', help='split to evaluate')
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()
    device = torch.device(args.device)
    print(device)

    base_model = args.base_model
    model_path = args.model_path

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    local_rank = int(os.environ.get("LOCAL_RANK") or 0)

    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    model = LlamaForCausalLM.from_pretrained(
            base_model,
            torch_dtype='auto',
            #device_map=device_map,
        )
    model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(
            model,
            model_path,
            torch_dtype='auto',
            #device_map=device_map,
        )

    tokenizer.padding_side = "left"

    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    print('tokenizer loaded from '+model_path)
    print('model loaded from '+model_path)

    model.to(device)
    model.eval()

    data_path = args.data_path

    sample_num = args.sample_num
    num_beams = args.num_beams

    valid_modes = args.split.split(',')

    for valid_mode in valid_modes:

        source_file = args.data_path+'/'+valid_mode+'.source'
        tgt_file = args.data_path+'/'+valid_mode+'.target'
        code_list = load_response(source_file, tgt_file)

        condidate_trie = Trie([tokenizer.encode(x)[1:] for x in code_list])
        print('\nTrie loaded, possible response count: ', len(condidate_trie))
        prefix_allowed_tokens = llama_prefix_allowed_tokens_fn(condidate_trie)

        recall_count_at_1 = 0
        recall_count_at_5 = 0
        recall_count_at_10 = 0

        total_inputs, total_labels = load_prompt(source_file, tgt_file, sub_size=sample_num)
        for item in tqdm(zip(total_inputs, total_labels), desc='Evaluating '+str(valid_mode)+' queries', total=len(total_inputs)):
            input, label = item
            input = tokenizer(input, return_tensors="pt")['input_ids']

            with torch.no_grad():

                generation_config = GenerationConfig(
                    num_beams=num_beams,
                    max_new_tokens=20,
                    num_return_sequences=num_beams,
                    early_stopping=True,
                    use_cache=False,
                )
                beams = model.generate(input.to(device),
                                        generation_config=generation_config,
                                        prefix_allowed_tokens_fn=prefix_allowed_tokens
                                        )
                beams = [tokenizer.decode(x, skip_special_tokens=True).strip().split('Response: ')[-1].split(' ') for x in beams]
                label = label.strip().split(' ')

                # print(beams)
                # print(label)
                hits = [i for i, x in enumerate(beams) if x == label]
                if len(hits) != 0:
                    recall_count_at_10 += 1
                    if hits[0] < 5:
                        recall_count_at_5 += 1
                    if hits[0] == 0:
                        recall_count_at_1 += 1
        

        recall_at_1 = recall_count_at_1 / len(total_inputs)
        recall_at_5 = recall_count_at_5 / len(total_inputs)
        recall_at_10 = recall_count_at_10 / len(total_inputs)

        print(valid_mode)
        print(f"Recall@1: {recall_at_1}")
        print(f"Recall@5: {recall_at_5}")
        print(f"Recall@10: {recall_at_10}") 

          

        