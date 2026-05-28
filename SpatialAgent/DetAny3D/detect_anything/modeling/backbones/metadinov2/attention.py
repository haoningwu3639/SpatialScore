# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging

import torch.nn as nn
from torch import Tensor

logger = logging.getLogger("dinov2")


try:
    from xformers.ops import fmha, memory_efficient_attention, unbind

    XFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("xFormers not available")
    XFORMERS_AVAILABLE = False


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )

        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None) -> Tensor:
        if not XFORMERS_AVAILABLE:
            assert attn_bias is None, "xFormers is required for nested tensors usage"
            return super().forward(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])
        # import matplotlib.pyplot as plt
        # import seaborn as sns
        # qkv = (
        #     self.qkv(x)
        #     .reshape(B, N, 3, self.num_heads, C // self.num_heads)
        #     .permute(2, 0, 3, 1, 4)
        # )
        # (q, k, v) = qkv[0] * self.scale, qkv[1], qkv[2]
        # attn = q @ k.transpose(-2, -1)
        # sim = attn[:, :, 0, 1:]
        # fig, axes = plt.subplots(4, 4, figsize=(12, 12))  # 创建一个 4x4 的网格
        # axes = axes.flatten()  # 将 axes 数组展平，方便索引
        # for i in range(16):
        #     ax = axes[i]
        #     # 画热力图，将 attn[i] 作为 (196, 196) 的矩阵绘制
        #     sns.heatmap(sim[0, i].view(-1, 64).detach().cpu().numpy(), cmap='viridis', annot=False, ax=ax)
        #     ax.set_title(f"Head {i + 1}")
        #     ax.axis('off')  # 关闭坐标轴
        # plt.tight_layout()
        # plt.savefig("unidepth_dino_last_4x4.png", dpi=300)
        # plt.close()
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
