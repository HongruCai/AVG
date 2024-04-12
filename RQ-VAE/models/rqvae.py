import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .layers import MLPLayers
from .rq import ResidualVectorQuantizer


class RQVAE(nn.Module):
    def __init__(self,
                 in_dim=768,
                 code_length=4,
                 # num_emb_list=[256,256,256,256],
                 num_emb_list=None,
                 e_dim=64,
                 # layers=[512,256,128],
                 layers=None,
                 dropout_prob=0.0,
                 bn=False,
                 loss_type="mse",
                 text_loss_type="mse",
                 text_loss_pos="before",
                 quant_loss_weight=1.0,
                 kmeans_init=False,
                 kmeans_iters=100,
                 # sk_epsilons=[0,0,0.003,0.01]],
        ):
        super(RQVAE, self).__init__()

        self.in_dim = in_dim
        self.num_emb_list = num_emb_list
        self.e_dim = e_dim

        self.layers = layers
        self.dropout_prob = dropout_prob
        self.bn = bn
        self.loss_type = loss_type
        self.text_loss_type = text_loss_type
        self.text_loss_pos = text_loss_pos

        self.quant_loss_weight=quant_loss_weight
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters

        self.encode_layer_dims = [self.in_dim] + self.layers + [self.e_dim]
        self.encoder = MLPLayers(layers=self.encode_layer_dims,
                                 dropout=self.dropout_prob,bn=self.bn)
        self.code_length = code_length
        self.rq = ResidualVectorQuantizer(num_emb_list, e_dim,
                                        code_length,
                                          kmeans_init = self.kmeans_init,
                                          kmeans_iters = self.kmeans_iters)

        self.decode_layer_dims = self.encode_layer_dims[::-1]
        self.decoder = MLPLayers(layers=self.decode_layer_dims,
                                       dropout=self.dropout_prob,bn=self.bn)

    def forward(self, x):
        x_encoded = self.encoder(x)
        x_q, rq_loss, indices = self.rq(x_encoded)
        out = self.decoder(x_q)

        return out, rq_loss, indices, x_encoded

    @torch.no_grad()
    def get_indices(self, xs):
        x_e = self.encoder(xs)
        _, _, indices = self.rq(x_e)
        return indices

    def compute_loss(self, out, encoder_out,quant_loss, xs, cap=None):

        if self.loss_type == 'mse':
            loss_recon = F.mse_loss(out, xs, reduction='mean')
        elif self.loss_type == 'l1':
            loss_recon = F.l1_loss(out, xs, reduction='mean')
        else:
            raise ValueError('incompatible loss type')
        
        if self.text_loss_pos == 'before':
            output = encoder_out
        elif self.text_loss_pos == 'after':
            output = out
            
        if self.text_loss_type == 'mse':
            loss_cap = F.mse_loss(output, cap, reduction='mean') if cap is not None else torch.tensor([0])
        elif self.text_loss_type == 'l1':
            loss_cap = F.l1_loss(output, cap, reduction='mean') if cap is not None else torch.tensor([0])
        elif self.text_loss_type == 'contrastive':
            if cap is not None:
                eps = 1e-12
                normalized_out = output / (output.norm(p=2, dim=1, keepdim=True) + eps)
                normalized_cap = cap / (cap.norm(p=2, dim=1, keepdim=True) + eps)
                logits_out = torch.matmul(normalized_out, normalized_cap.t())
                loss_1 =nn.functional.cross_entropy(logits_out, torch.arange(len(logits_out)).to(logits_out.device))
                loss_2 = nn.functional.cross_entropy(logits_out.t(), torch.arange(len(logits_out)).to(logits_out.device))
                loss_cap = (loss_1 + loss_2) / 2
            else:
                loss_cap = torch.tensor([0])

        loss_total = loss_recon + self.quant_loss_weight * quant_loss if cap is None else loss_cap + loss_recon + self.quant_loss_weight * quant_loss

        return loss_total, loss_recon, loss_cap