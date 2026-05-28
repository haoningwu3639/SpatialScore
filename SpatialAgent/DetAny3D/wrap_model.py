import torch.nn as nn
import torch.nn.functional as F
from detect_anything.datasets.utils import *
from detect_anything.modeling import ImageEncoderViT, MaskDecoder, PromptEncoder, TwoWayTransformer

from typing import Any, Dict, List, Tuple
from functools import partial

import torch
import random

def _build_sam(
    config=None,
):  
    checkpoint = config.model.checkpoint

    prompt_embed_dim = config.model.image_encoder.out_chans
    image_size = config.model.image_encoder.img_size
    vit_patch_size = config.model.image_encoder.patch_size
    image_embedding_size = image_size // vit_patch_size
    sam = Sam(
        image_encoder=ImageEncoderViT(
            depth=config.model.image_encoder.depth,
            embed_dim=config.model.image_encoder.embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=config.model.image_encoder.num_heads,
            patch_size=vit_patch_size,
            qkv_bias=config.model.image_encoder.qkv_bias,
            use_rel_pos=config.model.image_encoder.use_rel_pos,
            global_attn_indexes=config.model.image_encoder.global_attn_indexes,
            window_size=config.model.image_encoder.window_size,
            out_chans=prompt_embed_dim,
            cfg=config,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(config.model.pad, config.model.pad),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            cfg=config,
            num_multimask_outputs=config.model.mask_decoder.num_multimask_outputs,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
                inject_layer=1,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=config.dataset.pixel_mean,
        pixel_std=config.dataset.pixel_std,
    )
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        
        new_model_dict = sam.state_dict()
        for k,v in new_model_dict.items():
            if k in state_dict.keys():
                new_model_dict[k] = state_dict[k]
        sam.load_state_dict(new_model_dict)
    sam.mask_decoder.initzeroconv()
    return sam




class WrapModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.sam = _build_sam(
                config=cfg,
            )
        
    def setup(self):
        if self.cfg.model.freeze.image_encoder:
            for name, param in self.sam.image_encoder.named_parameters():
                param.requires_grad = False
        if self.cfg.model.freeze.prompt_encoder:
            for name, param in self.sam.prompt_encoder.named_parameters():
                param.requires_grad = False
        if self.cfg.model.freeze.mask_decoder:
            for name, param in self.sam.mask_decoder.named_parameters():
                param.requires_grad = False
                
    def forward(self, input_batch):
        ret_dict = {}
        images = input_batch['images']
        
        _, _, H, W = images.shape
        vit_H, vit_W = input_batch['vit_pad_size'][0]
        ret_dict.update(self.sam.image_encoder(input_batch))
        ret_dict['depth_maps'] = ret_dict['depth_maps'][:, 0, ...]

        if self.cfg.tune_with_prompt:
            image_embeddings = ret_dict['image_embeddings']
            
            metric_feature = ret_dict['metric_features']
            camera_feature = ret_dict['camera_features']
            depth_feature = ret_dict['depth_features'].permute(0, 3, 1, 2)

            _, _, height, width = depth_feature.shape

            # 确定长边和短边
            if height > width:
                target_height = max(vit_H, vit_W)
                target_width = int(width * (target_height / height))
            else:
                target_width = max(vit_H, vit_W)
                target_height = int(height * (target_width / width))

            depth_feature = F.interpolate(depth_feature, size=(target_height, target_width), mode='bilinear', align_corners=False)

            rays = ret_dict['rays']

            camera_feature = camera_feature.flatten(1).unsqueeze(1).unsqueeze(1).permute(0, 3, 1, 2).repeat(1, 1, depth_feature.size(2), depth_feature.size(3))
            metric_feature = metric_feature.flatten(1).unsqueeze(1).unsqueeze(1).permute(0, 3, 1, 2).repeat(1, 1, depth_feature.size(2), depth_feature.size(3))

            
            pad_height = self.cfg.model.pad // self.cfg.model.image_encoder.patch_size - depth_feature.size(2)  
            pad_width = self.cfg.model.pad // self.cfg.model.image_encoder.patch_size - depth_feature.size(3)  

            padding = (0, pad_width, 0, pad_height)


            depth_feature = F.pad(depth_feature, padding)
            camera_feature = F.pad(camera_feature, padding)
            metric_feature = F.pad(metric_feature, padding)

            point = None
            if input_batch.get('point_coords', None) is not None:
                point_coords = input_batch['point_coords']
                bs = point_coords.shape[0]
                num_points = point_coords.shape[1]
                label = torch.ones((bs, num_points)).to(point_coords.device).to(torch.int)
                point = (point_coords, label)
            elif input_batch.get('boxes_coords', None) is not None:
                bs = input_batch['boxes_coords'].shape[0]
            else:
                return ret_dict


            sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
                points=point,
                boxes=input_batch.get('boxes_coords', None),
                masks=None,
            )


            pos_embed = self.sam.prompt_encoder.get_dense_pe()
            h, w = image_embeddings.shape[-2], image_embeddings.shape[-1]
            
            pos_embed_interpolated = nn.functional.interpolate(
                pos_embed, 
                size=(h, w), 
                mode='bicubic',  
                align_corners=False
            )

            ret_dict.update(self.sam.mask_decoder(
                input_dict = ret_dict,
                image_embeddings=image_embeddings.repeat(bs, 1, 1, 1),
                # image_pe=self.sam.prompt_encoder.get_dense_pe()[..., :image_embeddings.shape[-2], :image_embeddings.shape[-1]], 
                image_pe=pos_embed_interpolated,
                metric_feature=metric_feature,
                camera_feature=camera_feature,
                depth_feature=depth_feature,
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True,
            ))

            
            ret_dict['masks'] = F.interpolate(
                ret_dict['masks'],
                (H, W),
                mode="bilinear",
                align_corners=False)
            ret_dict['pred_bbox_3d_depth'] = torch.clamp(ret_dict['pred_bbox_3d_depth'], max = np.log(self.cfg.max_depth))

        return ret_dict

class Sam(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        image_encoder,
        prompt_encoder,
        mask_decoder,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

    @property
    def device(self) -> Any:
        return self.pixel_mean.device