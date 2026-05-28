import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.models.layers import trunc_normal_

from math import pi
from typing import Optional

from .unidepth_utils import *


class ListAdapter(nn.Module):
    def __init__(self, input_dims, hidden_dim: int):
        super().__init__()
        self.input_adapters = nn.ModuleList([])
        self.num_chunks = len(input_dims)
        self.checkpoint = True
        for input_dim in input_dims:
            self.input_adapters.append(
                nn.Sequential(
                    nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim), nn.GELU()
                )
            )

    def forward(self, x: torch.Tensor, splits: torch.Tensor) -> torch.Tensor:
        xs = torch.split(x, splits.int().tolist(), dim=-1)
        xs = [adapter(x) for x, adapter in zip(xs, self.input_adapters)]
        return torch.cat(xs, dim=-1)

class CameraHead(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        expansion: int = 4,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.aggregate1 = AttentionBlock(
            hidden_dim, num_heads=1, expansion=expansion, dropout=dropout
        )
        self.aggregate2 = AttentionBlock(
            hidden_dim, num_heads=1, expansion=expansion, dropout=dropout
        )
        self.latents_pos = nn.Parameter(
            torch.randn(1, 4, hidden_dim), requires_grad=True
        )
        self.in_features = MLP(hidden_dim, expansion=2, dropout=dropout)
        self.project_cls = MLP(hidden_dim, dropout=dropout)
        self.out = MLP(hidden_dim, expansion=2, dropout=0.0, output_dim=1)

    def fill_intrinsics(self, x):
        camera_intrinsics = torch.zeros(
            x.shape[0], 3, 3, device=x.device, requires_grad=False
        )
        camera_intrinsics[:, 0, 0] = x[:, 0].exp()
        camera_intrinsics[:, 1, 1] = x[:, 1].exp()
        camera_intrinsics[:, 0, 2] = x[:, 2].sigmoid()
        camera_intrinsics[:, 1, 2] = x[:, 3].sigmoid()
        camera_intrinsics[:, 2, 2] = 1.0
        return camera_intrinsics

    def forward(self, features, cls_tokens, pos_embed) -> torch.Tensor:
        features = features.unbind(dim=-1)
        cls_tokens = self.project_cls(cls_tokens)
        latents_pos = self.latents_pos.expand(cls_tokens.shape[0], -1, -1)
        features = self.in_features(torch.cat(features, dim=1) + pos_embed)
        features = torch.cat((features, cls_tokens), dim=1)
        cls_tokens = self.aggregate1(
            cls_tokens, context=features, pos_embed=latents_pos
        )
        cls_tokens = self.aggregate2(
            cls_tokens, context=features, pos_embed=latents_pos
        )

        # project to intrinsics
        x = self.out(cls_tokens).squeeze(-1)
        camera_intrinsics = self.fill_intrinsics(x)

        return camera_intrinsics, cls_tokens

    def set_shapes(self, shapes):
        self.shapes = shapes


class GlobalHead(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        camera_dim: int,
        expansion: int = 4,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.camera_dim = camera_dim
        self.in_features = nn.Linear(hidden_dim, hidden_dim)
        self.project_rays = nn.Linear(camera_dim + 3, hidden_dim)
        self.aggregate1 = AttentionBlock(
            hidden_dim, num_heads=1, expansion=expansion, dropout=dropout
        )
        self.aggregate2 = AttentionBlock(
            hidden_dim, num_heads=1, expansion=expansion, dropout=dropout
        )
        self.project_cls = MLP(hidden_dim, dropout=dropout)
        self.out = MLP(hidden_dim, expansion=2, dropout=0.0, output_dim=1)

    def embed_rays(self, rays, shapes):
        rays_embedding = flat_interpolate(rays, old=self.original_shapes, new=shapes)
        rays_embedding = F.normalize(rays_embedding, dim=-1)
        rays_embedding = generate_fourier_features(
            rays_embedding,
            dim=self.camera_dim,
            max_freq=max(shapes) // 2,
            use_log=True,
            cat_orig=True,
        )
        return rays_embedding

    def set_original_shapes(self, shapes):
        self.original_shapes = shapes

    def set_shapes(self, shapes):
        self.shapes = shapes

    def get_scaleshift(self, x):
        scale, shift = torch.chunk(x, 2, dim=1)
        scale = scale.exp().reshape(-1, 1, 1, 1)
        # scale = scale.reshape(-1, 1, 1, 1)
        shift = shift.reshape(-1, 1, 1, 1)
        return scale, shift

    def forward(self, features, cls_tokens, rays) -> torch.Tensor:
        features = features.unbind(dim=-1)
        cls_tokens = self.project_cls(cls_tokens)
        rays_embedding = self.project_rays(self.embed_rays(rays, self.shapes))
        rays_embedding = rays_embedding.repeat(1, len(features), 1)
        features = self.in_features(torch.cat(features, dim=1) + rays_embedding)
        features = torch.cat((features, cls_tokens), dim=1)
        cls_tokens = self.aggregate1(cls_tokens, context=features)
        cls_tokens = self.aggregate2(cls_tokens, context=features)
        x = self.out(cls_tokens).squeeze(-1)
        scale, shift = self.get_scaleshift(x)
        return scale, shift, cls_tokens


class DepthHead(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        expansion: int = 4,
        depths = 4,
        checkpoint: bool = True,
        camera_dim: int = 256,
        num_resolutions: int = 4,
        dropout: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__()
        self.checkpoint = checkpoint
        self.camera_dim = camera_dim
        self.skip_depth = False

        self.to_latents = MLP(hidden_dim, expansion=2, dropout=dropout)
        self.features_channel_cat = nn.Linear(hidden_dim * num_resolutions, hidden_dim)
        self.aggregate_16 = AttentionBlock(
            hidden_dim,
            num_heads=1,
            expansion=expansion,
            dropout=dropout,
            context_dim=hidden_dim,
        )
        self.prompt_camera = AttentionBlock(
            hidden_dim,
            num_heads=1,
            expansion=expansion,
            dropout=dropout,
            context_dim=hidden_dim,
        )

        self.rays_layers = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        self.process_layers = nn.ModuleList([])
        self.norms, self.out_layers = nn.ModuleList([]), nn.ModuleList([])
        self.depth_mlp, self.confidence_mlp = nn.ModuleList([]), nn.ModuleList([])
        for i, depth in enumerate(depths):
            blk_lst = nn.ModuleList([])
            for _ in range(depth):
                blk_lst.append(
                    NystromBlock(
                        hidden_dim // int(2**i),
                        num_heads=num_heads // int(2**i),
                        expansion=expansion,
                        dropout=dropout,
                    )
                )
            self.process_layers.append(blk_lst)
            self.rays_layers.append(nn.Linear(camera_dim + 3, hidden_dim // int(2**i)))
            self.ups.append(
                ConvUpsampleShuffleResidual(
                    hidden_dim // int(2**i),
                    expansion=expansion,
                    kernel_size=7,
                    num_layers=2,
                )
            )
            self.depth_mlp.append(
                MLP(
                    input_dim=hidden_dim // int(2 ** (i + 1)),
                    output_dim=16,
                    expansion=1,
                )
            )
            self.confidence_mlp.append(
                MLP(
                    input_dim=hidden_dim // int(2 ** (i + 1)),
                    output_dim=16,
                    expansion=1,
                )
            )
        self.to_depth = nn.Conv2d(
            16 * len(depths), 1, 7, padding=3, padding_mode="reflect"
        )
        self.to_confidence = nn.Conv2d(
            16 * len(depths), 1, 7, padding=3, padding_mode="reflect"
        )

    def set_original_shapes(self, shapes):
        self.original_shapes = shapes

    def set_shapes(self, shapes):
        self.shapes = shapes

    def embed_rays(self, rays, shapes):
        rays_embedding = flat_interpolate(rays, old=self.original_shapes, new=shapes)
        rays_embedding = F.normalize(rays_embedding, dim=-1)
        rays_embedding = generate_fourier_features(
            rays_embedding,
            dim=self.camera_dim,
            max_freq=max(shapes) // 2,
            use_log=True,
            cat_orig=True,
        )
        return rays_embedding

    def project_rays(self, rays, shapes):
        embedded_rays = []
        for i, layer in enumerate(self.rays_layers):
            embedded_rays.append(
                layer(self.embed_rays(rays, [(2**i) * x for x in shapes]))
            )
        return embedded_rays

    def decode_depth(self, latents_16, rays, shapes):
        latents = latents_16
        out_features, depths, confidences = [], [], []
        for i, (up, layers, rays_embedding) in enumerate(
            zip(self.ups, self.process_layers, rays)
        ):
            for layer in layers:
                latents = layer(latents, pos_embed=rays_embedding)
            latents = up(
                rearrange(
                    latents + rays_embedding,
                    "b (h w) c -> b c h w",
                    h=shapes[0] * int(2**i),
                    w=shapes[1] * int(2**i),
                ).contiguous()
            )
            out = rearrange(
                latents,
                "b (h w) c -> b h w c",
                h=shapes[0] * int(2 ** (1 + i)),
                w=shapes[1] * int(2 ** (1 + i)),
            )
            out_features.append(out)

        # aggregate output and project to depth
        for i, (layer, features) in enumerate(
            zip(self.depth_mlp[::-1], out_features[::-1])
        ):
            out_depth_features = layer(features).permute(0, 3, 1, 2)
            out_depth_features = F.interpolate(
                out_depth_features, size=self.original_shapes, mode="bilinear"
            )
            depths.append(out_depth_features)
        logdepth = self.to_depth(torch.cat(depths, dim=1))

        # aggregate output and project to confidences
        for i, (layer, features) in enumerate(
            zip(self.confidence_mlp[::-1], out_features[::-1])
        ):
            out_conf_features = layer(features).permute(0, 3, 1, 2)
            out_conf_features = F.interpolate(
                out_conf_features, size=self.original_shapes, mode="bilinear"
            )
            confidences.append(out_conf_features)
        confidence = self.to_confidence(torch.cat(confidences, dim=1))

        # apply sigmoid ot get conf in [0, 1]
        confidence = torch.sigmoid(confidence)

        return logdepth, confidence

    def init_latents(self, features, shapes):
        # Generate latents with init as pooled features
        features_channels = torch.cat(features, dim=-1)
        features_16 = self.features_channel_cat(features_channels)
        latents_16 = features_16 + self.to_latents(
            flat_interpolate(features_16, old=self.shapes, new=shapes, antialias=False)
        )
        return latents_16

    def forward(
        self, features: torch.Tensor, rays_hr: torch.Tensor, pos_embed, level_embed
    ) -> torch.Tensor:
        B = features.shape[0]
        features = features.unbind(dim=-1)
        shapes = self.shapes #(47, 64)
        # camera_embedding
        rays_embeddings = self.project_rays(rays_hr, shapes)
        # Init latents
        init_latents_16 = self.init_latents(features, shapes) #torch.Size([1, 3008, 640])
        # Aggregate features: F -> D
        latents_16_0 = self.aggregate_16(
            init_latents_16,
            context=torch.cat(features, dim=1),
            pos_embed_context=pos_embed + level_embed,
        )
        # Aggregate camera: D -> D|E
        latents_16 = self.prompt_camera(latents_16_0, context=rays_embeddings[0])
        # Decode depth
        logdepth, confidence = self.decode_depth(latents_16, rays_embeddings, shapes)

        return logdepth, confidence, latents_16, latents_16_0


class Unidepth_Decoder(nn.Module):
    def __init__(
        self,
        config,
    ):
        super().__init__()
        self.build(config)
        self.cfg = config
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def get_adapted_features_sam(self, features_flat, splits):
        features_flat_cat = torch.cat(features_flat, dim=-1)
        features_projected = self.input_adapter_for_sam(
            features_flat_cat, splits
        )  # list [b hw c] shapes
        features = torch.chunk(features_projected, splits.shape[0], dim=-1)
        return features
    def get_adapted_features(self, features_flat, splits):
        features_flat_cat = torch.cat(features_flat, dim=-1)
        features_projected = self.input_adapter(
            features_flat_cat, splits
        )  # list [b hw c] shapes
        features = torch.chunk(features_projected, splits.shape[0], dim=-1)
        return features

    def run_camera(self, cls_tokens, features, pos_embed, original_shapes, rays_gt):

        # camera layer
        intrinsics, cls_tokens = self.camera_layer(
            features=features, cls_tokens=cls_tokens, pos_embed=pos_embed
        )
    
        intrinsics[:, 0, 0] = max(original_shapes) / 2 * intrinsics[:, 0, 0]
        intrinsics[:, 1, 1] = max(original_shapes) / 2 * intrinsics[:, 1, 1]
        intrinsics[:, 0, 2] = intrinsics[:, 0, 2] * original_shapes[1]
        intrinsics[:, 1, 2] = intrinsics[:, 1, 2] * original_shapes[0]

        rays = (
            rays_gt
            if rays_gt is not None
            else generate_rays(intrinsics, original_shapes)[0]
        )
        return intrinsics, rays, cls_tokens

    def run_global(self, cls_tokens, features, rays):
        # get cls tokens projections

        scale, shift, metric_feature = self.global_layer(
            features=features, rays=rays, cls_tokens=cls_tokens
        )

        return scale, shift, metric_feature

    def forward(self, input_dict):

        B, C, H, W = input_dict["image_for_dino"].shape
        device = input_dict["image_for_dino"].device
        dtype = input_dict["image_for_dino"].dtype

        if input_dict.get('gt_intrinsic') is not None:
            rays, angles = generate_rays(input_dict['gt_intrinsic'].to(dtype), (H, W))
            input_dict["rays"] = rays
            input_dict["angles"] = angles
            input_dict["gt_K"] = input_dict['gt_intrinsic'].to(dtype)

        global_tokens_dino = [input_dict['dino_token'][i] for i in [-2, -1]]
        camera_tokens_dino = [input_dict['dino_token'][i] for i in [-3, -2, -1, -2]]
        features_dino = [input_dict['dino_feature'][i] for i in range(4)]
        
        patch_shape_H, patch_shape_W = int(input_dict['vit_pad_size'][0, 0]), int(input_dict['vit_pad_size'][0, 1])
        # 将 dino_features 插值到 (patch_shape_H, patch_shape_W)
        features_dino = [
            F.interpolate(feature.permute(0, 3, 1, 2), size=(patch_shape_H, patch_shape_W), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
            for feature in features_dino
        ]

        level_shapes = sorted(
        list(set([tuple([x.shape[1], x.shape[2]]) for x in features_dino]))
        )[::-1]

        if len(level_shapes) == 1:
            level_shapes = level_shapes * self.num_resolutions
        
        input_shapes = level_shapes
        common_shape = level_shapes[-2]
        patch_H, patch_W = common_shape

        # input shapes repeat shapes for each level, times the amount of the layers:
        features_flat_dino = [
            flat_interpolate(
                rearrange(x, "b h w c -> b (h w) c"), old=input_shape, new=common_shape
            )
            for x, input_shape in zip(features_dino, input_shapes)
        ]
        features_splits = torch.tensor(
            [x.shape[-1] for x in features_flat_dino],
            device=device,
            requires_grad=False,
            dtype=torch.float32,
        )

        features_dino = self.get_adapted_features(features_flat_dino, features_splits)
        features_dino = torch.stack(features_dino, dim=-1)

        # get cls tokens projections
        cls_tokens_splits = torch.tensor(
            [x.shape[-1] for x in camera_tokens_dino],
            device=features_dino.device,
            requires_grad=False,
            dtype=features_dino.dtype,
        )
        camera_tokens_dino = torch.cat(camera_tokens_dino, dim=-1)
        camera_tokens_dino = self.camera_token_adapter(camera_tokens_dino, cls_tokens_splits)
        camera_tokens_dino = torch.cat(
            torch.chunk(camera_tokens_dino, cls_tokens_splits.shape[0], dim=-1), dim=1
        )

        cls_tokens_splits = torch.tensor(
            [x.shape[-1] for x in global_tokens_dino],
            device=features_dino.device,
            requires_grad=False,
            dtype=torch.float32,
        )
        global_tokens_dino = torch.cat(global_tokens_dino, dim=-1)
        global_tokens_dino = self.global_token_adapter(global_tokens_dino, cls_tokens_splits)
        global_tokens_dino = torch.cat(
            torch.chunk(global_tokens_dino, cls_tokens_splits.shape[0], dim=-1), dim=1
        )

        
        device = input_dict["images"].device
        dtype = input_dict["images"].dtype
        global_tokens_sam = [input_dict['cls_token'][i] for i in [-2, -1]]
        camera_tokens_sam = [input_dict['cls_token'][i] for i in [-3, -2, -1, -2]]
        features_sam = [input_dict['multi_features'][i][2].permute(0, 2, 3, 1) for i in range(4)]

        level_shapes_sam = sorted(
            list(set([tuple([x.shape[1], x.shape[2]]) for x in features_sam]))
        )[::-1]

        if len(level_shapes_sam) == 1:
            level_shapes_sam = level_shapes_sam * self.num_resolutions
        
        input_shapes = [[patch_shape_H, patch_shape_W], [patch_shape_H, patch_shape_W], [patch_shape_H, patch_shape_W], [patch_shape_H, patch_shape_W],]
        common_shape = level_shapes_sam[-2]

        # input shapes repeat shapes for each level, times the amount of the layers:
        features_flat_sam = [
            flat_interpolate(
                rearrange(x, "b h w c -> b (h w) c"), old=input_shape, new=common_shape
            )
            for x, input_shape in zip(features_sam, input_shapes)
        ]
        features_splits_sam = torch.tensor(
            [x.shape[-1] for x in features_flat_sam],
            device=device,
            requires_grad=False,
            dtype=torch.float32,
        )

        features_sam = self.get_adapted_features_sam(features_flat_sam, features_splits_sam)
        features_sam = torch.stack(features_sam, dim=-1)

        # get cls tokens projections
        cls_tokens_splits_sam = torch.tensor(
            [x.shape[-1] for x in camera_tokens_sam],
            device=features_sam.device,
            requires_grad=False,
            dtype=features_sam.dtype,
        )
        camera_tokens_sam = torch.cat(camera_tokens_sam, dim=-1)
        camera_tokens_sam = self.camera_token_adapter_for_sam(camera_tokens_sam, cls_tokens_splits_sam)
        camera_tokens_sam = torch.cat(
            torch.chunk(camera_tokens_sam, cls_tokens_splits_sam.shape[0], dim=-1), dim=1
        )

        cls_tokens_splits_sam = torch.tensor(
            [x.shape[-1] for x in global_tokens_sam],
            device=features_sam.device,
            requires_grad=False,
            dtype=torch.float32,
        )
        global_tokens_sam = torch.cat(global_tokens_sam, dim=-1)
        global_tokens_sam = self.global_token_adapter_for_sam(global_tokens_sam, cls_tokens_splits_sam)
        global_tokens_sam = torch.cat(
            torch.chunk(global_tokens_sam, cls_tokens_splits_sam.shape[0], dim=-1), dim=1
        )

        global_tokens = global_tokens_sam + global_tokens_dino
        camera_tokens = camera_tokens_sam + camera_tokens_dino
        features = features_sam + features_dino


        # positional embeddings, spatial and level
        level_embed = torch.cat(
            [
                self.level_embed_layer(self.level_embeds)[i : i + 1]
                .unsqueeze(0)
                .repeat(B, common_shape[0] * common_shape[1], 1)
                for i in range(self.num_resolutions)
            ],
            dim=1,
        )
        dummy_tensor = torch.zeros(
            B, 1, common_shape[0], common_shape[1], device=device, requires_grad=False
        )
        pos_embed = self.pos_embed(dummy_tensor)
        pos_embed = rearrange(pos_embed, "b c h w -> b (h w) c").repeat(
            1, self.num_resolutions, 1
        )
  
        self.camera_layer.set_shapes(common_shape)
        intrinsics, rays, camera_feature = self.run_camera(
            camera_tokens,
            features=features,
            pos_embed=pos_embed + level_embed,
            original_shapes=(H, W),
            rays_gt=input_dict.get("rays"),
        )

        self.global_layer.set_shapes(common_shape)
        self.global_layer.set_original_shapes((H, W))
        scale, shift, metric_feature = self.run_global(
            global_tokens, features=features, rays=rays
        )

        # run bulk of the model
        self.depth_layer.set_shapes(common_shape)
        self.depth_layer.set_original_shapes((H, W))
        logdepth, confidence, depth_features, depth_features0 = self.depth_layer(
            features=features,
            rays_hr=rays,
            pos_embed=pos_embed,
            level_embed=level_embed,
        )
        logdepth = logdepth.to(torch.float32, non_blocking=True)

        # norm in log space, why performs better?
        shapes = [int(x) for x in logdepth.shape[-2:]]
        # depth_normalized = F.layer_norm(logdepth, shapes)
        depth_normalized = F.layer_norm(logdepth, shapes).exp()

        depth = (
            depth_normalized + shift
        ) * scale  # shift is scale invariant if we do (x + mu) * sigma
        depth = F.softplus(depth, beta=10.0).to(dtype, non_blocking=True)

        outputs = {
            "depth_maps": depth,
            "confidence": confidence,
            "depth_features": depth_features.reshape(B, patch_H, patch_W, -1).contiguous(),
            "metric_features": metric_feature,
            "camera_features": camera_feature,
            "scale": scale,
            "shift": shift,
            "pred_K": intrinsics,
            "rays": rays,
        }
        return outputs


    
    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"latents_pos", "level_embeds"}

    def build(self, cfg):
        input_dims = [1280, 1280, 1280, 1280]
        hidden_dim = 512
        input_dims_for_dino = [1024, 1024, 1024, 1024]
        expansion = 4
        num_heads = 8
        dropout = 0.0
        depth = [6, 0, 0]
        depths_encoder = [8, 16, 24, 32]
        self.downsample = 4
        self.num_resolutions = 4

        self.slices_encoder = [[0, 7], [8, 15], [16, 23], [24, 31]]
        cls_token_input_dims = [1280, 1280, 1280, 1280]

        # # camera layer
        self.camera_layer = CameraHead(
            hidden_dim=hidden_dim,
            num_heads=8,
            expansion=4,
            dropout=0.0,
        )

        # # scale shift layer
        self.global_layer = GlobalHead(
            hidden_dim=hidden_dim,
            camera_dim=96,
            num_heads=8,
            expansion=4,
            dropout=0.0,
        )

        # # adapt from encoder features, just project

        self.camera_token_adapter = ListAdapter(input_dims_for_dino, hidden_dim)
        self.global_token_adapter = ListAdapter(input_dims_for_dino[:2], hidden_dim)
        self.input_adapter = ListAdapter(input_dims_for_dino, hidden_dim)
        self.input_adapter_for_sam = ListAdapter(cls_token_input_dims, hidden_dim)
        self.camera_token_adapter_for_sam = ListAdapter(cls_token_input_dims, hidden_dim)
        self.global_token_adapter_for_sam = ListAdapter(cls_token_input_dims[:2], hidden_dim)


        self.depth_layer = DepthHead(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            expansion=expansion,
            depths=depth,
            dropout=dropout,
            camera_dim=96,
            num_resolutions=self.num_resolutions,
        )

        self.pos_embed = PositionEmbeddingSine(hidden_dim // 2, normalize=True)
        self.level_embeds = nn.Parameter(
            torch.randn(len(input_dims), hidden_dim), requires_grad=True
        )
        self.level_embed_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        

