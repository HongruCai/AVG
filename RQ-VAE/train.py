import argparse
import random
import torch
import numpy as np
from time import time
import logging

from torch.utils.data import DataLoader

from dataset import EmbDataset,CapEmbDataset,CapPseudoEmbDataset
from models.rqvae import RQVAE
from trainer import  Trainer
import wandb
import os
from utils import ensure_dir, set_color, get_local_time


def parse_args():
    parser = argparse.ArgumentParser(description="RQ-VAE")

    parser.add_argument('--version', type=str, default='rq-vae', help='version')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--epochs', type=int, default=2000, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=2048, help='batch size')
    parser.add_argument('--num_workers', type=int, default=4, )
    parser.add_argument('--eval_step', type=int, default=200, help='eval step')
    parser.add_argument('--save_step', type=int, default=500, help='save step')
    parser.add_argument('--learner', type=str, default="AdamW", help='optimizer')

    parser.add_argument('--weight_decay', type=float, default=0, help='l2 regularization weight')
    parser.add_argument("--dropout_prob", type=float, default=0.25, help="dropout ratio")
    parser.add_argument("--bn", action='store_true', help="use bn or not")
    parser.add_argument("--loss_type", type=str, default="mse", help="loss_type")
    parser.add_argument("--kmeans_init", action='store_true', help="use kmeans_init or not")
    parser.add_argument("--kmeans_iters", type=int, default=100, help="max kmeans iters")

    parser.add_argument("--device", type=str, default="cuda", help="gpu or cpu")

    parser.add_argument('--num_emb_list', type=int, nargs='+', default=[1024], help='emb num of every rq')
    parser.add_argument('--code_length', type=int, default=4, help='rq code length')
    parser.add_argument('--e_dim', type=int, default=768, help='rq codebook embedding size')
    parser.add_argument('--quant_loss_weight', type=float, default=1.0, help='rq quantion loss weight')
    parser.add_argument('--layers', type=int, nargs='+', default=[1024,512], help='hidden sizes of every layer')

    parser.add_argument("--dataset", type=str, default="flickr", help="dataset name")
    parser.add_argument("--ckpt_dir", type=str, default="./RQ-VAE/output/rqvae/", help="output directory for model")
    parser.add_argument("--use_cap", action='store_true', help="if use caption for reconstruction loss")
    parser.add_argument("--use_pseudo", action='store_true', help="if use pseudo caption")
    parser.add_argument("--text_loss_type", type=str, default="mse", help="text loss type")
    parser.add_argument("--text_loss_pos", type=str, default="after", help="text loss position")

    return parser.parse_args()


if __name__ == '__main__':
    seed = 2024
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args = parse_args()
    print(args)

    version = args.version
    loss_type = args.text_loss_type[:3]
    sturcture = 'VT' if args.use_cap else 'V'
    mlp_layers = str(args.layers[0])+'-'+str(args.layers[-1])
    codebook_num = str(len(args.num_emb_list))
    codebook = 'c'+str(args.num_emb_list[0])
    epochs = 'e'+str(args.epochs)
    lr = 'lr'+str(args.lr)
    name = version+'_'+sturcture+'_'+mlp_layers+'_'+codebook_num+'-'+codebook+'_'+epochs+'_'+lr+'_'+loss_type
    print(name)
    
    wandb.init(project="rq-vae", config=args, name=name)
    
    dataset = args.dataset
    save_dir = args.ckpt_dir + '/' + name
    os.makedirs(save_dir)
    logging.basicConfig(filename=save_dir+'/training.log', level=logging.INFO, format='%(asctime)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.log(logging.INFO, args)

    if args.use_cap:
        if args.use_pseudo:
            train_data = CapPseudoEmbDataset(img_path='data/'+dataset+'/emb/train_emb_images.npy',
                                             cap_path='data/'+dataset+'/emb/train_emb_captions.npy',
                                             pseudo_img='data/'+dataset+'/emb/pseudo_emb_images.npy',
                                             pseudo_cap='data/'+dataset+'/emb/pseudo_emb_captions.npy')
            valid_data = CapEmbDataset(img_path='data/'+dataset+'/emb/val_emb_images.npy',
                                       cap_path='data/'+dataset+'/emb/val_emb_captions.npy')
        else:
            train_data = CapEmbDataset(img_path='data/'+dataset+'/emb/train_emb_images.npy',
                                       cap_path='data/'+dataset+'/emb/train_emb_captions.npy')
            valid_data = CapEmbDataset(img_path='data/'+dataset+'/emb/val_emb_images.npy',
                                       cap_path='data/'+dataset+'/emb/val_emb_captions.npy')
    else:
        train_data = EmbDataset(data_path="data/"+dataset+"/emb/train_emb_images.npy")
        valid_data = EmbDataset(data_path="data/"+dataset+"/emb/val_emb_images.npy")

    print("num of training data",len(train_data))
    print("num of valid data",len(valid_data))
    
    model = RQVAE(in_dim=train_data.dim,
                num_emb_list=args.num_emb_list,
                code_length=args.code_length,
                e_dim=args.e_dim,
                layers=args.layers,
                dropout_prob=args.dropout_prob,
                bn=args.bn,
                loss_type=args.loss_type,
                text_loss_type=args.text_loss_type,
                text_loss_pos=args.text_loss_pos,
                quant_loss_weight=args.quant_loss_weight,
                kmeans_init=args.kmeans_init,
                kmeans_iters=args.kmeans_iters,
                
                )
    
    print(model)
    train_data_loader = DataLoader(train_data,num_workers=args.num_workers,
                             batch_size=args.batch_size, shuffle=False,
                             pin_memory=True)
    valid_data_loader = DataLoader(valid_data,num_workers=args.num_workers,
                                batch_size=args.batch_size, shuffle=True,
                                pin_memory=True)
    trainer = Trainer(args,model,wandb,save_dir=save_dir,logger=logger)
    best_val_loss = trainer.fit(train_data=train_data_loader,valid_data=valid_data_loader)


