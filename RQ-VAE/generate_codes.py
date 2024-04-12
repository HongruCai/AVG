import argparse
import random
import torch
import numpy as np
from time import time
import logging

from torch.utils.data import DataLoader,Dataset

from dataset import FilteredEmbDataset
from models.rqvae import RQVAE
import os
from tqdm import tqdm
import json


def parse_args():
    parser = argparse.ArgumentParser(description="Generate codes from RQ-VAE and save codebook")

    parser.add_argument('--ckpt_path', type=str, default='t2v/output/rqvae/', help='path to the model checkpoint')
    parser.add_argument('--dataset', type=str, default='flickr', help='dataset name')

    return parser.parse_args()


if __name__ == '__main__':
    generate_args = parse_args()
    ckpt_path = generate_args.ckpt_path
    dataset = generate_args.dataset
    
    device = torch.device("cuda")
    ckpt = torch.load(ckpt_path, map_location=device)
    args = ckpt["args"]
    state_dict = ckpt["state_dict"]

    model = RQVAE(in_dim=768,
                code_length=args.code_length,
                num_emb_list=args.num_emb_list,
                e_dim=args.e_dim,
                layers=args.layers,
                dropout_prob=args.dropout_prob,
                bn=args.bn,
                loss_type=args.loss_type,
                quant_loss_weight=args.quant_loss_weight,
                kmeans_init=args.kmeans_init,
                kmeans_iters=args.kmeans_iters,
                )

    model.load_state_dict(state_dict)

    rq_embedding = []
    for i in range(len(model.rq.vq_layers)):
        rq_embedding.append(model.rq.vq_layers[i].embedding.weight.data)

    rq_embedding = torch.cat(rq_embedding, dim=0)

    print(rq_embedding.shape)
    print('save rq_embedding to file:', os.path.dirname(ckpt_path)+'/codebook_embedding.pt')
    torch.save(rq_embedding, os.path.dirname(ckpt_path)+'/codebook_embedding.pt')

    model = model.to(device)
    model.eval()
    print(model)

    split = ["train","val","test"]

    with open("data/"+dataset+"/"+dataset+"_split_captions.json") as f:
        split_cap = json.load(f)

    rec = {}
    for s in split:
        embeddings_path = "data/"+dataset+f"/emb/{s}_emb_images.npy"
        img_id_path = "data/"+dataset+f"/emb/{s}_emb.json"

        with open(img_id_path) as f:
            img_id = json.load(f)
        indices = list(range(0, len(img_id), 5))
        #print(indices)
    
        filtered_data = FilteredEmbDataset(data_path=embeddings_path, indices=indices)
        dataloader = DataLoader(filtered_data, batch_size=1, shuffle=False, num_workers=args.num_workers)
        
        print("num of", s, "data", len(filtered_data))

        codes = []
        for i, batch in tqdm(enumerate(dataloader),desc=f"Processing {s} data",total=len(dataloader)):
            x = batch
            with torch.no_grad():
                real_idx = indices[i]  
                codes = model.get_indices(x.to(device))
                codes = codes.view(-1, codes.shape[-1]).cpu().numpy()
                real_img_id = img_id[real_idx]  
                rec[real_img_id] = {
                    "code": codes.tolist()[0],
                    #"code": [f"{prefix[i]}{c}" for i, c in enumerate(codes.tolist()[0])],
                    "split": split_cap[real_img_id]['split'] if 'split' in split_cap[real_img_id] else None,
                    "caption": split_cap[real_img_id]['caption']
                }

    with open(os.path.dirname(ckpt_path)+'/'+dataset+'_codes.json', "w") as f:
        json.dump(rec, f, indent=4)