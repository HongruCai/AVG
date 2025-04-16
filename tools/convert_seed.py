import json
import hydra
from omegaconf import OmegaConf
from PIL import Image
import pyrootutils
import os
from tqdm import tqdm
import argparse
import torch
import sys

def parse_args():
    parser = argparse.ArgumentParser(description="Encode images and append tokens to metadata JSON.")
    parser.add_argument('--image_caption_file', type=str, required=True,
                        help='Input JSON file containing image names and captions')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Output path for the updated JSON')
    parser.add_argument('--image_dirs', nargs='+', required=True,
                        help='List of image root directories to search for images')
    parser.add_argument('--tokenizer_cfg', type=str, required=True,
                        help='Path to tokenizer config YAML')
    parser.add_argument('--transform_cfg', type=str, required=True,
                        help='Path to transform config YAML')
    parser.add_argument('--device', type=str, default='cuda', help='Device for inference')
    return parser.parse_args()


def main():
    pyrootutils.setup_root(__file__, indicator='.project-root', pythonpath=True)
    args = parse_args()

    tokenizer_cfg = OmegaConf.load(args.tokenizer_cfg)
    tokenizer = hydra.utils.instantiate(tokenizer_cfg, device=args.device, load_diffusion=False)
    tokenizer.to(args.device) if hasattr(tokenizer, 'to') else None

    transform_cfg = OmegaConf.load(args.transform_cfg)
    transform = hydra.utils.instantiate(transform_cfg)

    with open(args.image_caption_file, 'r') as f:
        image_caption_data = json.load(f)

    avg2seed = {}
    for image_name, data in tqdm(image_caption_data.items()):
        image_loaded = False
        for base_dir in args.image_dirs:
            image_path = os.path.join(base_dir, image_name)
            if os.path.exists(image_path):
                try:
                    image = Image.open(image_path).convert('RGB')
                    image_tensor = transform(image).unsqueeze(0).to(args.device)
                    with torch.no_grad():
                        image_ids = tokenizer.encode_image(image_torch=image_tensor)
                    avg_id = ' '.join(['c_'+str(i) for i in data['code']])
                    avg2seed[avg_id] = image_ids.tolist()[0]
                    image_loaded = True
                    break
                except Exception as e:
                    print(f"Error in {image_name}: {e}")
                    break
        if not image_loaded:
            print(f"Image {image_name} not found。")
        else:
            with open(args.output_file, 'w') as f:
                json.dump(avg2seed, f)
    print(f"Encoded images and saved to {args.output_file}")

if __name__ == '__main__':
    main()
