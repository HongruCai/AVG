import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import json
from dataset import ImgCapEmbDataset
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Prepare embeddings for RQ-VAE training")

    parser.add_argument('--root_dir', type=str, default='data/flickr/images/split', help='root directory of images')
    parser.add_argument('--caption_file', type=str, default='data/flickr/flickr_split_captions.json', help='caption file')
    parser.add_argument('--pseudo_caption_file', type=str, default='data/flickr/pseudo_query.json', help='pseudo caption file')
    parser.add_argument('--clip_model', type=str, default='openai/clip-vit-large-patch14-336', help='CLIP model')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--save_path', type=str, default='data/flickr/emb', help='save path')

    return parser.parse_args()


def process_and_save_embeddings(root_dir, caption, clip_model='openai/clip-vit-large-patch14-336', batch_size=1024, save_path='embeddings.pt',pseudo=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = ImgCapEmbDataset(root_dir=root_dir,caption_file=caption, clip_model=clip_model, device=device, pseudo_file=pseudo)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    img_embeddings = []
    cap_embeddings = []
    rec = []
    for img_emb,cap_emb,img_name in tqdm(dataloader):
        img_embeddings.append(img_emb.cpu().numpy()) 
        cap_embeddings.append(cap_emb.cpu().numpy())
        rec.extend(img_name)
    img_embeddings_np = np.concatenate(img_embeddings, axis=0)
    cap_embeddings_np = np.concatenate(cap_embeddings, axis=0)
    print(img_embeddings_np.shape)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path+'_images.npy', img_embeddings_np)
    np.save(save_path+'_captions.npy', cap_embeddings_np)
    #print(len(rec))
    with open(save_path+'.json', 'w') as fp:
        json.dump(rec,fp,indent=4)


if __name__ == '__main__':
    args = parse_args()
    print(args)
    
    for split in ['test','val','train']:
        process_and_save_embeddings(root_dir=args.root_dir+'/'+split+'/', 
                                    caption=args.caption_file, 
                                    clip_model=args.clip_model, 
                                    batch_size=args.batch_size, 
                                    save_path=args.save_path+'/'+split+'_emb',
                                    pseudo=False)
    
    # generate pseudo query embeddings
    process_and_save_embeddings(root_dir=args.root_dir+'/'+'test'+'/', 
                                caption=args.pseudo_caption_file, 
                                clip_model=args.clip_model, 
                                batch_size=args.batch_size, 
                                save_path=args.save_path+'/'+'pseudo_emb',
                                pseudo=True)
 