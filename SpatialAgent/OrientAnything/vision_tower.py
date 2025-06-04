import torch
from torch import nn
import torch.nn.init as init
import torch.nn.functional as F

from typing import Dict, List, Optional, Set, Tuple, Union
from transformers import AutoImageProcessor, AutoModel, Dinov2Model
from transformers.models.dinov2.modeling_dinov2 import Dinov2Embeddings
from transformers.models.dinov2.configuration_dinov2 import Dinov2Config
import numpy as np
from contextlib import nullcontext

DINO_SMALL  = "facebook/dinov2-small"
DINO_BASE   = "facebook/dinov2-base"
DINO_LARGE  = "/remote-home/haoningwu/SpatialAgent/checkpoints/dinov2-large"
DINO_GIANT  = "facebook/dinov2-giant"


def get_activation(activation):
    if activation.lower() == 'gelu':
        return nn.GELU()
    elif activation.lower() == 'rrelu':
        return nn.RReLU(inplace=True)
    elif activation.lower() == 'selu':
        return nn.SELU(inplace=True)
    elif activation.lower() == 'silu':
        return nn.SiLU(inplace=True)
    elif activation.lower() == 'hardswish':
        return nn.Hardswish(inplace=True)
    elif activation.lower() == 'leakyrelu':
        return nn.LeakyReLU(inplace=True)
    elif activation.lower() == 'sigmoid':
        return nn.Sigmoid()
    elif activation.lower() == 'tanh':
        return nn.Tanh()
    else:
        return nn.ReLU(inplace=True)



class MLP_dim(nn.Module):
    def __init__(
        self, in_dim=512, out_dim=1024, bias=True, activation='relu'):
        super().__init__()
        self.act = get_activation(activation)
        self.net1 = nn.Sequential(
            nn.Linear(in_dim, int(out_dim), bias=bias),
            nn.BatchNorm1d(int(out_dim)),
            self.act
        )
        self.net2 = nn.Sequential(
            nn.Linear(int(out_dim), out_dim, bias=bias),
            nn.BatchNorm1d(out_dim)
        )

    def forward(self, x):
        return self.net2(self.net1(x))

class FLIP_Dinov2Embeddings(Dinov2Embeddings):
    """
    Construct the CLS token, mask token, position and patch embeddings.
    """

    def __init__(self, config: Dinov2Config) -> None:
        super().__init__(config)

    def forward(self, pixel_values: torch.Tensor, bool_masked_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, _, height, width = pixel_values.shape
        target_dtype = self.patch_embeddings.projection.weight.dtype
        embeddings = self.patch_embeddings(pixel_values.to(dtype=target_dtype))

        # add the [CLS] token to the embedded patch tokens
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        # add positional encoding to each token
        embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)

        if bool_masked_pos is not None:
            # embeddings = torch.where(
            #     bool_masked_pos.unsqueeze(-1), self.mask_token.to(embeddings.dtype).unsqueeze(0), embeddings
            # )
            B,S,D = embeddings.shape
            batch_indices = torch.arange(B).unsqueeze(1)
            embeddings = embeddings[batch_indices, bool_masked_pos]

        embeddings = self.dropout(embeddings)

        return embeddings

class FLIP_DINOv2(Dinov2Model):
    def __init__(self, config):
        super().__init__(config)
        
        self.embeddings = FLIP_Dinov2Embeddings(config)
        
class DINOv2_MLP(nn.Module):
    def __init__(self, 
                 dino_mode,
                 in_dim,
                 out_dim,
                 evaluate,
                 mask_dino,
                 frozen_back
                ) -> None:
        super().__init__()
        # self.dinov2 = AutoModel.from_pretrained(DINO_BASE)
        if dino_mode == 'base':
            self.dinov2 = FLIP_DINOv2.from_pretrained(DINO_BASE, cache_dir='./')
        elif dino_mode == 'large':
            self.dinov2 = FLIP_DINOv2.from_pretrained(DINO_LARGE, cache_dir='./')
        elif dino_mode == 'small':
            self.dinov2 = FLIP_DINOv2.from_pretrained(DINO_SMALL, cache_dir='./')
        elif dino_mode == 'giant':
            self.dinov2 = FLIP_DINOv2.from_pretrained(DINO_GIANT, cache_dir='./')
        
        self.down_sampler = MLP_dim(in_dim=in_dim, out_dim=out_dim)
        self.random_mask  = False
        if not evaluate:
            self.init_weights(self.down_sampler)
            self.random_mask = mask_dino
        if frozen_back:
            self.forward_mode = torch.no_grad()
        else:
            self.forward_mode = nullcontext()
        
    def forward(self, img_inputs):
        device = self.get_device()
        # print(img_inputs['pixel_values'].shape)
        
        with self.forward_mode:
            if self.random_mask:
                B = len(img_inputs['pixel_values'])
                S = 256
                indices = []
                for i in range(B):
                    tmp = torch.randperm(S)[:S//2]
                    tmp = tmp.sort().values + 1
                    indices.append(tmp)
                indices = torch.stack(indices, dim=0)
                indices = torch.cat([torch.zeros(B, 1, dtype=torch.long, device='cpu'), indices], dim=1)
                # print(indices.shape)
                img_inputs['bool_masked_pos'] = indices.to(device)
                
            dino_outputs = self.dinov2(**img_inputs)
            dino_seq = dino_outputs.last_hidden_state
            # B,S,_ = dino_seq.shape
            # dino_seq = dino_seq.view(B*S,-1)
            dino_seq = dino_seq[:,0,:]
        
        down_sample_out = self.down_sampler(dino_seq)
        # down_sample_out = down_sample_out.view(B,S,-1)
        # down_sample_out = down_sample_out[:,0,:]
        
        return down_sample_out
    
    def get_device(self):
        return next(self.parameters()).device
    
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)

