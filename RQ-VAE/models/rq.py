import torch
import torch.nn as nn

from .vq import VectorQuantizer


class ResidualVectorQuantizer(nn.Module):
    """ References:
        SoundStream: An End-to-End Neural Audio Codec
        https://arxiv.org/pdf/2107.03312.pdf
    """

    def __init__(self, n_e_list, e_dim, code_length,
                 kmeans_init = False, kmeans_iters = 100):
        super().__init__()
        self.n_e_list = n_e_list
        self.e_dim = e_dim
        self.code_length = code_length
        self.num_quantizers = len(n_e_list)
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters

        self.vq_layers = nn.ModuleList([VectorQuantizer(n_e, e_dim,
                                                        kmeans_init = self.kmeans_init,
                                                        kmeans_iters = self.kmeans_iters,
                                                        )
                                        for n_e in n_e_list ])

    def get_codebook(self):
        all_codebook = []
        for quantizer in self.vq_layers:
            codebook = quantizer.get_codebook()
            all_codebook.append(codebook)
        return torch.stack(all_codebook)

    def forward(self, x):
        all_losses = []
        all_indices = []

        x_q = 0
        residual = x
        if self.num_quantizers == 1:
            quantizer = self.vq_layers[0]
            for _ in range(self.code_length):
                x_res, loss, indices = quantizer(residual)
                residual = residual - x_res
                x_q = x_q + x_res

                all_losses.append(loss)
                all_indices.append(indices)
        else:
            for quantizer in self.vq_layers:
                x_res, loss, indices = quantizer(residual)
                residual = residual - x_res
                x_q = x_q + x_res

                all_losses.append(loss)
                all_indices.append(indices)

        mean_losses = torch.stack(all_losses).mean()
        all_indices = torch.stack(all_indices, dim=-1)

        return x_q, mean_losses, all_indices