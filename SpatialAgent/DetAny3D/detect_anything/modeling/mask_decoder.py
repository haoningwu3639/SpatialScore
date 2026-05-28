# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment

from typing import List, Tuple, Type
import numpy as np

from .common import LayerNorm2d
import copy
from detect_anything.datasets.utils import (
    box3d_iou,
    angle2class,
    class2angle,
    points_img2cam,
    draw_bbox_2d,
    compute_3d_bbox_vertices_batch,
    rotation_6d_to_matrix)


class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        cfg,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()

        self.transformer_dim = transformer_dim
        self.iou_head_hidden_dim = iou_head_hidden_dim
        self.iou_head_depth = iou_head_depth
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )

        # sam original head
        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )
        self.cfg = cfg
        assert self.cfg.model.multi_level_box_output == 1 or self.cfg.model.original_sam == False, "original sam can only achieve when multi_level_box_output must be 1"

        if self.cfg.model.multi_level_box_output > 1:
            self.init_multi_level_box_layers()
        else:
            # bbox tokens
            self.box_token = nn.Embedding(1, transformer_dim)
            self.box_token_3d = nn.Embedding(1, transformer_dim)

            # bbox related heads: 2d bbox + 3d center projected to 2d
            self.bbox_head = MLP(
                transformer_dim, iou_head_hidden_dim, 6, iou_head_depth + 1
            )
            # self.bbox_3d_center_head = MLP(
            #     transformer_dim, iou_head_hidden_dim, 2, iou_head_depth
            # )
            self.bbox_3d_depth_head = MLP(
                transformer_dim, iou_head_hidden_dim, 2, iou_head_depth
            )
            self.bbox_3d_dims_head = MLP(
                transformer_dim, iou_head_hidden_dim, 3, iou_head_depth
            )

            if cfg.output_rotation_matrix:
                self.bbox_3d_rotation_matrix_out = MLP(
                    transformer_dim, iou_head_hidden_dim, 6, iou_head_depth
                )

            self.bbox_3d_alpha_cls_head = MLP(
                transformer_dim, iou_head_hidden_dim, 24, iou_head_depth
            )

        self.zero_conv2d_cam = nn.Conv2d(2048, 256, 1)
        self.zero_conv2d_metric = nn.Conv2d(1024, 256, 1)
        self.zero_conv2d = nn.Conv2d(512, 256, 1)
        
        self.out_zero_conv2d = nn.Conv2d(256, 256, 1)

    def initzeroconv(self):
        self.transformer2 = copy.deepcopy(self.transformer)
        self.transformer2.is_copy = True

    def init_multi_level_box_layers(self):
        self.box_iou_token = nn.Embedding(1, self.transformer_dim)
        self.num_box_tokens = self.cfg.model.multi_level_box_output
        self.box_token = nn.Embedding(self.num_box_tokens, self.transformer_dim)
        self.box_token_3d = nn.Embedding(self.num_box_tokens, self.transformer_dim)

        # Initialize output heads for each bbox
        self.bbox_heads = nn.ModuleList(
            [
                MLP(self.transformer_dim, self.iou_head_hidden_dim, 6, self.iou_head_depth)  # 2D bbox
                for _ in range(self.num_box_tokens)
            ]
        )

        # 3D bounding box related heads: 3D depth, dimensions, and alpha_cls for each box
        self.bbox_3d_depth_heads = nn.ModuleList(
            [
                MLP(self.transformer_dim, self.iou_head_hidden_dim, 2, self.iou_head_depth)
                for _ in range(self.num_box_tokens)
            ]
        )
        self.bbox_3d_dims_heads = nn.ModuleList(
            [
                MLP(self.transformer_dim, self.iou_head_hidden_dim, 3, self.iou_head_depth)
                for _ in range(self.num_box_tokens)
            ]
        )
        if self.cfg.output_rotation_matrix:
            self.bbox_3d_rotation_matrix_heads = nn.ModuleList(
                [
                    MLP(self.transformer_dim, self.iou_head_hidden_dim, 6, self.iou_head_depth)
                    for _ in range(self.num_box_tokens)
                ]
            )

        self.bbox_3d_alpha_cls_heads = nn.ModuleList(
            [
                MLP(self.transformer_dim, self.iou_head_hidden_dim, 24, self.iou_head_depth)
                for _ in range(self.num_box_tokens)
            ]
        )

        # Prediction mechanism: classify which box is the best
        self.bbox_iou_prediction_head = MLP(
            self.transformer_dim, self.iou_head_hidden_dim, self.num_box_tokens, self.iou_head_depth
        )
        
    def forward(
        self,
        input_dict,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        metric_feature,
        camera_feature,
        depth_feature,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """
        if self.cfg.model.multi_level_box_output > 1:
            ret_dict = self.predict_masks_multi_level(
                input_dict = input_dict,
                image_embeddings=image_embeddings,
                image_pe=image_pe,
                metric_feature=metric_feature,
                camera_feature=camera_feature,
                depth_feature=depth_feature,
                sparse_prompt_embeddings=sparse_prompt_embeddings,
                dense_prompt_embeddings=dense_prompt_embeddings,
            )

        else:
            ret_dict = self.predict_masks(
                image_embeddings=image_embeddings,
                image_pe=image_pe,
                metric_feature=metric_feature,
                camera_feature=camera_feature,
                depth_feature=depth_feature,
                sparse_prompt_embeddings=sparse_prompt_embeddings,
                dense_prompt_embeddings=dense_prompt_embeddings,
            )

        # Select the correct mask or masks for output
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)

        ret_dict['masks'] = ret_dict['masks'][:, mask_slice, :, :]
        ret_dict['iou_predictions'] = ret_dict['iou_predictions'][:, mask_slice]

        # Prepare output
        return ret_dict

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        metric_feature,
        camera_feature,
        depth_feature,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        # import ipdb;ipdb.set_trace()
        
        
        metric_embeddings = self.zero_conv2d_metric(metric_feature)
        camera_embeddings = self.zero_conv2d_cam(camera_feature)
        depth_embeddings = self.zero_conv2d(depth_feature)

        if self.cfg.model.original_sam:
            output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        else:
            output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight, self.box_token.weight, self.box_token_3d.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)

        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)
        
        # for zero_conv
        control_tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        
        h, w = image_embeddings.shape[-2], image_embeddings.shape[-1]
        
        dense_prompt_embeddings_interpolated = nn.functional.interpolate(
            dense_prompt_embeddings, 
            size=(h, w), 
            mode='bicubic',  
            align_corners=False
        )
            
        src = image_embeddings + dense_prompt_embeddings_interpolated
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape 

        control_q, control_k = self.transformer2(src + (depth_embeddings + metric_embeddings + camera_embeddings).repeat(b, 1, 1, 1), pos_src, control_tokens)
        control_k = self.out_zero_conv2d(control_k.reshape(-1, self.cfg.model.pad // self.cfg.model.image_encoder.patch_size, self.cfg.model.pad // self.cfg.model.image_encoder.patch_size, 256).permute(0, 3, 1, 2)).flatten(2).permute(0, 2, 1)
        hs, src = self.transformer(src, pos_src, tokens, control_q, control_k)

        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]
        if self.cfg.model.original_sam:
            box_tokens_out = iou_token_out
            box_3d_tokens_out = iou_token_out
        else:
            box_tokens_out = hs[:, 1 + self.num_mask_tokens, :]
            box_3d_tokens_out = hs[:, 2 + self.num_mask_tokens, :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        hyper_in_list: List[torch.Tensor] = []
        
        for i in range(self.num_mask_tokens):
            # with torch.no_grad():
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        
        hyper_in = torch.stack(hyper_in_list, dim=1)

        #  2d bbox
        if not self.cfg.contain_edge_obj:
            pred_bbox_2d = self.bbox_head(box_tokens_out).sigmoid()
        else:
            pred_bbox_2d = self.bbox_head(box_tokens_out)
        # 3d bbox
        # pred_center_2d = self.bbox_3d_center_head(box_3d_tokens_out)
        pred_bbox_3d_depth_hs = self.bbox_3d_depth_head(box_3d_tokens_out)
        pred_bbox_3d_dims = self.bbox_3d_dims_head(box_3d_tokens_out)
        
        pred_bbox_3d_alpha_hs = self.bbox_3d_alpha_cls_head(box_3d_tokens_out)

        pred_center_2d = pred_bbox_2d[..., :2]
        
        pred_bbox_2d = pred_bbox_2d[..., 2:]

        pred_bbox_3d_depth = pred_bbox_3d_depth_hs[..., :1]
        pred_bbox_3d_depth_log_variance = pred_bbox_3d_depth_hs[..., 1:]
        
        pred_bbox_3d_alpha_cls = pred_bbox_3d_alpha_hs[..., :12]
        pred_bbox_3d_alpha_res = pred_bbox_3d_alpha_hs[..., 12:]

        # output sam masks
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)
        iou_pred = self.iou_prediction_head(iou_token_out)

        pred_pose_6d = None
        if self.cfg.output_rotation_matrix:
            pred_pose_6d = self.bbox_3d_rotation_matrix_out(box_3d_tokens_out)

        return {
            "masks": masks,
            "iou_predictions": iou_pred,
            "pred_bbox_2d": pred_bbox_2d,
            "pred_center_2d": pred_center_2d,
            "pred_bbox_3d_depth": pred_bbox_3d_depth,
            "pred_bbox_3d_depth_log_variance": pred_bbox_3d_depth_log_variance,
            "pred_bbox_3d_dims": pred_bbox_3d_dims,
            "pred_bbox_3d_alpha_cls": pred_bbox_3d_alpha_cls,
            "pred_bbox_3d_alpha_res": pred_bbox_3d_alpha_res,
            "pred_pose_6d": pred_pose_6d,
            "pred_box_ious": None,
            "box_iou_loss": None
        }
    
    def predict_masks_multi_level(
        self,
        input_dict,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        metric_feature,
        camera_feature,
        depth_feature,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        metric_embeddings = self.zero_conv2d_metric(metric_feature)
        camera_embeddings = self.zero_conv2d_cam(camera_feature)
        depth_embeddings = self.zero_conv2d(depth_feature)
        
        output_tokens = torch.cat([self.iou_token.weight, self.box_iou_token.weight, self.mask_tokens.weight, self.box_token.weight, self.box_token_3d.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)

        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)
       
        control_tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        h, w = image_embeddings.shape[-2], image_embeddings.shape[-1]
        
        dense_prompt_embeddings_interpolated = nn.functional.interpolate(
            dense_prompt_embeddings, 
            size=(h, w), 
            mode='bicubic',  
            align_corners=False
        )

        src = image_embeddings + dense_prompt_embeddings_interpolated
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape 

        
        control_q, control_k = self.transformer2(src + (depth_embeddings + metric_embeddings + camera_embeddings).repeat(b, 1, 1, 1), pos_src, control_tokens)
        control_k = self.out_zero_conv2d(control_k.reshape(-1, self.cfg.model.pad // self.cfg.model.image_encoder.patch_size, self.cfg.model.pad // self.cfg.model.image_encoder.patch_size, 256).permute(0, 3, 1, 2)).flatten(2).permute(0, 2, 1)
        hs, src = self.transformer(src, pos_src, tokens, control_q, control_k)

        iou_token_out = hs[:, 0, :]
        box_iou_token_out = hs[:, 1, :]
        mask_tokens_out = hs[:, 2 : (2 + self.num_mask_tokens), :]
        box_tokens_out = hs[:, 2 + self.num_mask_tokens : 2 + self.num_mask_tokens + self.num_box_tokens, :]
        box_3d_tokens_out = hs[:, 2 + self.num_mask_tokens + self.num_box_tokens : 2 + self.num_mask_tokens + 2 * self.num_box_tokens, :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        hyper_in_list: List[torch.Tensor] = []
        
        for i in range(self.num_mask_tokens):
            # with torch.no_grad():
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        
        hyper_in = torch.stack(hyper_in_list, dim=1)
        # output sam masks
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)
        iou_pred = self.iou_prediction_head(iou_token_out)

        pred_bbox_2d_list = []
        pred_bbox_3d_depth_list = []
        pred_bbox_3d_dims_list = []
        pred_bbox_3d_alpha_cls_list = []
        pred_pose_6d_list = []

        for i in range(self.num_box_tokens):

            #  2d bbox
            if not self.cfg.contain_edge_obj:
                pred_bbox_2d_list.append(self.bbox_heads[i](box_tokens_out[:, i, :]).sigmoid())
            else:
                pred_bbox_2d_list.append(self.bbox_heads[i](box_tokens_out[:, i, :]))
            
            pred_bbox_3d_depth_list.append(self.bbox_3d_depth_heads[i](box_3d_tokens_out[:, i, :]))
            pred_bbox_3d_dims_list.append(self.bbox_3d_dims_heads[i](box_3d_tokens_out[:, i, :]))
            pred_bbox_3d_alpha_cls_list.append(self.bbox_3d_alpha_cls_heads[i](box_3d_tokens_out[:, i, :]))
            
            if self.cfg.output_rotation_matrix:
                pred_pose_6d_list.append(self.bbox_3d_rotation_matrix_heads[i](box_3d_tokens_out[:, i, :]))
        
        if self.training:
            
            bbox_iou_list = []
            for i in range(self.num_box_tokens):
                
                pred_center_2d = pred_bbox_2d_list[i][..., :2]
                pred_bbox_2d = pred_bbox_2d_list[i][..., 2:]
                pred_bbox_3d_depth = pred_bbox_3d_depth_list[i][..., :1]
                pred_bbox_3d_dims = pred_bbox_3d_dims_list[i][..., :3]
                pred_bbox_3d_alpha_cls = pred_bbox_3d_alpha_cls_list[i][..., :12]
                pred_bbox_3d_alpha_res = pred_bbox_3d_alpha_cls_list[i][..., 12:]
                if self.cfg.output_rotation_matrix:
                    pred_pose_6d = rotation_6d_to_matrix(pred_pose_6d_list[i])
                else:
                    pred_pose_6d = None
                
                K_for_convert = input_dict.get('gt_intrinsic', input_dict['pred_K']).to(pred_center_2d.device)

                decoded_pred_center_2d = pred_center_2d * self.cfg.model.pad

                bboxes_pred_2d_center_x = decoded_pred_center_2d[..., 0]
                bboxes_pred_2d_center_y = decoded_pred_center_2d[..., 1]

                pred_alpha_cls = torch.argmax(pred_bbox_3d_alpha_cls, dim=-1)
                gt_angle_cls = input_dict.get('gt_angle_cls', None)
                pred_alpha_res = pred_bbox_3d_alpha_res[np.arange(gt_angle_cls.shape[0]), gt_angle_cls]
                pred_alpha = class2angle(pred_alpha_cls, pred_alpha_res)

                pred_ry = torch.atan2(
                    bboxes_pred_2d_center_x - K_for_convert[..., 0, 2], K_for_convert[..., 0, 0]) + pred_alpha
                pred_ry[pred_ry > torch.pi] = pred_ry[pred_ry > torch.pi] - torch.pi * 2
                pred_ry[pred_ry < -torch.pi] = pred_ry[pred_ry < -torch.pi] + torch.pi * 2
                pred_ry = pred_ry.unsqueeze(-1)

                pred_centers_2d_with_depth = torch.cat([decoded_pred_center_2d, pred_bbox_3d_depth.exp()], dim=-1)
                pred_centers_3d = points_img2cam(pred_centers_2d_with_depth, K_for_convert[0])
                decoded_bboxes_pred_3d = torch.cat([pred_centers_3d, pred_bbox_3d_dims.exp(), pred_ry], dim = -1)

                # TODO: sth wrong with calculate 3d iou when pitch and roll is not zero, so we do not calculate 3d bbox with rot mat here
                decoded_bboxes_pred_3d_corners = compute_3d_bbox_vertices_batch(decoded_bboxes_pred_3d)
                decoded_bboxes_gt_3d_corners = compute_3d_bbox_vertices_batch(input_dict['gt_bboxes_3d'])
                
                with torch.no_grad():
                    bbox_iou_list.append([])
                    for j in range(decoded_bboxes_pred_3d_corners.shape[0]):
                        todo_pred_corners = decoded_bboxes_pred_3d_corners[j].detach().cpu().numpy()
                        todo_gt_corners = decoded_bboxes_gt_3d_corners[j].detach().cpu().numpy()
                        bbox_iou_list[i].append(box3d_iou(todo_pred_corners, todo_gt_corners))
            
            
            bbox_iou_tensor = torch.tensor(bbox_iou_list, device=pred_center_2d.device).transpose(0, 1).float().clamp(0, 1)
            pred_box_ious = self.bbox_iou_prediction_head(box_iou_token_out).sigmoid()
            box_iou_loss = F.mse_loss(pred_box_ious, bbox_iou_tensor, reduction='none').sum() / bbox_iou_tensor.shape[0]

            max_iou_values, max_iou_indices = torch.max(bbox_iou_tensor, dim=1)

            pred_bbox_2d_tensor = torch.stack(
                [pred_bbox_2d_list[i][j] for i, j in zip(max_iou_indices, range(pred_bbox_2d_list[0].shape[0]))],
                dim=0
            )
            pred_bbox_3d_depth_tensor = torch.stack(
                [pred_bbox_3d_depth_list[i][j] for i, j in zip(max_iou_indices, range(pred_bbox_3d_depth_list[0].shape[0]))],
                dim=0
            )
            pred_bbox_3d_dims_tensor = torch.stack(
                [pred_bbox_3d_dims_list[i][j] for i, j in zip(max_iou_indices, range(pred_bbox_3d_dims_list[0].shape[0]))],
                dim=0
            )
            pred_bbox_3d_alpha_cls_tensor = torch.stack(
                [pred_bbox_3d_alpha_cls_list[i][j] for i, j in zip(max_iou_indices, range(pred_bbox_3d_alpha_cls_list[0].shape[0]))],
                dim=0
            )
            if self.cfg.output_rotation_matrix:
                pred_pose_6d_tensor = torch.stack(
                    [pred_pose_6d_list[i][j] for i, j in zip(max_iou_indices, range(pred_pose_6d_list[0].shape[0]))],
                    dim=0
                )
            else:
                pred_pose_6d_tensor = None
            
            pred_center_2d = pred_bbox_2d_tensor[..., :2]
            pred_bbox_2d = pred_bbox_2d_tensor[..., 2:]
            # if self.cfg.contain_edge_obj:
            #     pred_bbox_2d = pred_bbox_2d_tensor[..., 2:].sigmoid()
            # else:
            pred_bbox_2d = pred_bbox_2d_tensor[..., 2:]

            pred_bbox_3d_depth = pred_bbox_3d_depth_tensor[..., :1]
            pred_bbox_3d_depth_log_variance = pred_bbox_3d_depth_tensor[..., 1:]
            
            pred_bbox_3d_alpha_cls = pred_bbox_3d_alpha_cls_tensor[..., :12]
            pred_bbox_3d_alpha_res = pred_bbox_3d_alpha_cls_tensor[..., 12:]
        
        else:
            # import ipdb; ipdb.set_trace()
            pred_box_ious = self.bbox_iou_prediction_head(box_iou_token_out).sigmoid()
            max_iou_values, max_iou_indices = torch.max(pred_box_ious, dim=1)
            
            pred_bbox_2d_tensor = torch.stack(
                [pred_bbox_2d_list[i][j] for i, j in zip(max_iou_indices, range(pred_bbox_2d_list[0].shape[0]))],
                dim=0
            )
            pred_bbox_3d_depth_tensor = torch.stack(
                [pred_bbox_3d_depth_list[i][j] for i, j in zip(max_iou_indices, range(pred_bbox_3d_depth_list[0].shape[0]))],
                dim=0
            )
            pred_bbox_3d_dims_tensor = torch.stack(
                [pred_bbox_3d_dims_list[i][j] for i, j in zip(max_iou_indices, range(pred_bbox_3d_dims_list[0].shape[0]))],
                dim=0
            )
            pred_bbox_3d_alpha_cls_tensor = torch.stack(
                [pred_bbox_3d_alpha_cls_list[i][j] for i, j in zip(max_iou_indices, range(pred_bbox_3d_alpha_cls_list[0].shape[0]))],
                dim=0
            )
            if self.cfg.output_rotation_matrix:
                pred_pose_6d_tensor = torch.stack(
                    [pred_pose_6d_list[i][j] for i, j in zip(max_iou_indices, range(pred_pose_6d_list[0].shape[0]))],
                    dim=0
                )
            else:
                pred_pose_6d_tensor = None
            
            pred_center_2d = pred_bbox_2d_tensor[..., :2]
            pred_bbox_2d = pred_bbox_2d_tensor[..., 2:]

            pred_bbox_3d_depth = pred_bbox_3d_depth_tensor[..., :1]
            pred_bbox_3d_depth_log_variance = pred_bbox_3d_depth_tensor[..., 1:]
            
            pred_bbox_3d_alpha_cls = pred_bbox_3d_alpha_cls_tensor[..., :12]
            pred_bbox_3d_alpha_res = pred_bbox_3d_alpha_cls_tensor[..., 12:]

            box_iou_loss = None

        pred_bbox_2d_tensor_all = torch.stack(pred_bbox_2d_list, dim=0).detach()
        pred_bbox_3d_depth_tensor_all = torch.stack(pred_bbox_3d_depth_list, dim=0).detach()
        pred_bbox_3d_dims_tensor_all = torch.stack(pred_bbox_3d_dims_list, dim=0).detach()
        pred_bbox_3d_alpha_cls_tensor_all = torch.stack(pred_bbox_3d_alpha_cls_list, dim=0).detach()
        if self.cfg.output_rotation_matrix:
            pred_pose_6d_tensor_all = torch.stack(pred_pose_6d_list, dim=0).detach()
        else:
            pred_pose_6d_tensor_all = None

        return {
            "masks": masks,
            "iou_predictions": iou_pred,
            "pred_bbox_2d": pred_bbox_2d,
            "pred_center_2d": pred_center_2d,
            "pred_bbox_3d_depth": pred_bbox_3d_depth,
            "pred_bbox_3d_depth_log_variance": pred_bbox_3d_depth_log_variance,
            "pred_bbox_3d_dims": pred_bbox_3d_dims_tensor,
            "pred_bbox_3d_alpha_cls": pred_bbox_3d_alpha_cls,
            "pred_bbox_3d_alpha_res": pred_bbox_3d_alpha_res,
            "pred_pose_6d": pred_pose_6d_tensor,
            "pred_box_ious": pred_box_ious,
            "box_iou_loss": box_iou_loss,
            "pred_bbox_2d_tensor_all": pred_bbox_2d_tensor_all,
            "pred_bbox_3d_depth_tensor_all": pred_bbox_3d_depth_tensor_all,
            "pred_bbox_3d_dims_tensor_all": pred_bbox_3d_dims_tensor_all,
            "pred_bbox_3d_alpha_cls_tensor_all": pred_bbox_3d_alpha_cls_tensor_all,
            "pred_pose_6d_tensor_all": pred_pose_6d_tensor_all,
        }


# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
            
        if self.sigmoid_output:
            x = x.sigmoid()
        return x

def select_best_bbox(pred_bboxes, gt_bboxes):
    """
    对每个 object 选择与 GT bbox IoU 最大的 bbox
    :param pred_bboxes: (3, num_objects, 7) 预测的3D边界框
    :param gt_bboxes: (1, num_objects, 7) GT 3D边界框
    :return: (1, num_objects, 7) 与 GT 重叠最大的 bbox
    """
    num_candidates, num_objects, _ = pred_bboxes.shape
    pred_bboxes = pred_bboxes.permute(1, 0, 2)  # 转为 (num_objects, 3, 7)
    gt_bboxes = gt_bboxes.squeeze(0)  # 转为 (num_objects, 7)

    # 扩展维度用于广播计算 IoU
    pred_bboxes_exp = pred_bboxes.unsqueeze(2)  # (num_objects, 3, 1, 7)
    gt_bboxes_exp = gt_bboxes.unsqueeze(1)  # (num_objects, 1, 7)

    # 计算 IoU，结果为 (num_objects, 3)
    ious = compute_iou_3d(
        pred_bboxes_exp.view(-1, 7), 
        gt_bboxes_exp.expand(-1, num_candidates, -1).reshape(-1, 7)
    ).view(num_objects, num_candidates)

    # 找到 IoU 最大的索引
    best_idx = torch.argmax(ious, dim=1)  # (num_objects,)

    # 根据索引选取对应的 bbox
    best_bboxes = pred_bboxes[torch.arange(num_objects), best_idx]  # (num_objects, 7)

    return best_bboxes.unsqueeze(0)  # (1, num_objects, 7)
