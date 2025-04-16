import collections
import logging
import numpy as np
import torch
import time
from torch import optim
from tqdm import tqdm

import base64
import openai
import json
import requests
from openai import OpenAI
import os
import sys
from tqdm import tqdm
import random
from transformers import CLIPProcessor, CLIPModel
import datasets as ds
import numpy as np
import json
import os
import dashscope
import tempfile
from http import HTTPStatus


from dashscope import MultiModalConversation
dashscope.api_key = "..."

system_prompt = '''
You are an expert at analyzing images. 
Given the example image and its corresponding captions, generate five captions for a new image that reflect the style and format of the examples. 
The description should start with the main object in the picture as the subject.
Avoid using summaries or general language or emotionally charged words. Use objective, descriptive, plain statements.
'''

system_prompt_no_ref = '''
You are an expert at analyzing images. 
Given the example captions, generate five captions for the image that reflect the style and format of the examples. 
The description should start with the main object in the picture as the subject.
Avoid using summaries or general language or emotionally charged words. Use objective, descriptive, plain statements.
'''


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def find_similar_image(img):
    with open('RQ-VAE/data/flickr/emb/test_emb.json', 'r') as f:
        test_emb_list = json.load(f)
    with open('RQ-VAE/data/flickr/emb/train_emb.json', 'r') as f:
        train_emb_list = json.load(f)
    test_emb = np.load('RQ-VAE/data/flickr/emb/test_emb_images.npy')
    train_emb = np.load('RQ-VAE/data/flickr/emb/train_emb_images.npy')

    img_emb = test_emb[test_emb_list.index(img), :]
    sim = np.dot(train_emb, img_emb)/(np.linalg.norm(train_emb, axis=1)*np.linalg.norm(img_emb))

    return train_emb_list[np.argmax(sim)]


def generate_pseudo_query_gpt(ori_image, ref_img, ref_cap):
    api_key = '...'
    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content":  system_prompt,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{ref_img}"}
                    },
                    {
                        "type": "text",
                        "text": ref_cap,
                    },
                    {
                        "type": "text",
                        "text": "Generate captions for the following image in a similar style.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{ori_image}"}
                    },
                ],
            }
        ],
        max_tokens=300,
    )

    caption = response.choices[0].message.content

    return caption



if __name__ == '__main__':

    with open('data/flickr/pseudo_query.json', 'r') as f:
        pseudo_query = json.load(f)

    all_images = [x['image_id'] for x in pseudo_query]

    if os.path.exists('data/flickr/pseudo_query_qwen.json'):
        rec = json.load(open('data/flickr/pseudo_query_qwen.json'))
    else:
        rec = []

    with open('RQ-VAE/data/flickr/flickr_split_captions.json', 'r') as f:
        raw_caps = json.load(f)

    pseudo_query = [x for x in pseudo_query if x['image_id'] not in [y['image_id'] for y in rec]]
    for image in tqdm(pseudo_query):
        image_id = image['image_id']
        # Get the full path of the image from the dataset

        image_file_path = image_id

        ref_image_path = find_similar_image(image_file_path)

        ref_cap = raw_caps[ref_image_path]['caption']
        ref_cap = "Example captions:\n"+"\n".join(f"{i+1}. {x}" for i, x in enumerate(ref_cap))
        # print(ref_cap)

        ref_img = f'data/flickr/flickr30k/Images/{ref_image_path}'
        ori_image = f'data/flickr/flickr30k/Images/{image_file_path}'
        # print(ori_image)
        caption = generate_pseudo_query_gpt(encode_image(ori_image), [], ref_cap)
        # print(caption)
        # sys.exit()
        if caption == []:
            continue

        rec.append({
            'image_id': image_id,
            'caption': caption,
        })

        with open('data/flickr/pseudo_query.json', 'w') as fp:
            json.dump(rec, fp, indent=4)

    with open('data/flickr/pseudo_query.json', 'w') as fp:
        json.dump(rec, fp, indent=4)
