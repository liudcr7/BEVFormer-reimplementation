"""PerceptionTransformer for BEVFormer, aligned with original implementation."""
from typing import Optional, List, Tuple
import numpy as np
import torch
import torch.nn as nn

try:
    from mmdet.models.utils.builder import TRANSFORMER
    from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
    from mmcv.cnn import xavier_init
    from mmcv.runner.base_module import BaseModule
    from mmcv.runner import auto_fp16
    from torch.nn.init import normal_
    from torchvision.transforms.functional import rotate
except Exception:
    BaseModule = nn.Module
    TRANSFORMER = None
    def build_transformer_layer_sequence(cfg): return None
    def xavier_init(module, *args, **kwargs): pass
    def normal_(tensor, *args, **kwargs): pass
    def auto_fp16(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    def rotate(img, angle, center): return img  # fallback

@TRANSFORMER.register_module()
class PerceptionTransformer(BaseModule):
    """PerceptionTransformer for BEVFormer.
    
    This module coordinates:
    - Building BEV queries
    - Temporal self-attention in BEV space
    - Spatial cross-attention that samples multi-view image features
    - Decoder for object detection
    
    The implementation is aligned with the original BEVFormer.
    """
    def __init__(
        self,
        embed_dims: int = 256,
        num_feature_levels: int = 4,
        num_cams: int = 6,
        encoder: Optional[dict] = None,
        decoder: Optional[dict] = None,
        positional_encoding: Optional[dict] = None,
        rotate_prev_bev: bool = True,
        use_shift: bool = True,
        use_can_bus: bool = True,
        can_bus_norm: bool = True,
        use_cams_embeds: bool = True,
        rotate_center: List[int] = [100, 100],
        **kwargs,
    ) -> None:
        super().__init__()
        self.embed_dims = embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.rotate_prev_bev = rotate_prev_bev
        self.use_shift = use_shift
        self.use_can_bus = use_can_bus
        self.can_bus_norm = can_bus_norm
        self.use_cams_embeds = use_cams_embeds
        self.rotate_center = rotate_center
        self.fp16_enabled = False
        
        # Build encoder and decoder
        if encoder is not None:
            self.encoder = build_transformer_layer_sequence(encoder)
        else:
            self.encoder = None
        
        if decoder is not None:
            self.decoder = build_transformer_layer_sequence(decoder)
        else:
            self.decoder = None
        
        self.init_layers()
    
    def init_layers(self):
        """Initialize layers of the PerceptionTransformer."""
        # Level embeddings for multi-scale features
        self.level_embeds = nn.Parameter(torch.Tensor(
            self.num_feature_levels, self.embed_dims))
        
        # Camera embeddings
        self.cams_embeds = nn.Parameter(
            torch.Tensor(self.num_cams, self.embed_dims))
        
        # Reference points for decoder
        self.reference_points = nn.Linear(self.embed_dims, 3)
        
        # CAN bus MLP
        self.can_bus_mlp = nn.Sequential(
            nn.Linear(18, self.embed_dims // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims // 2, self.embed_dims),
            nn.ReLU(inplace=True),
        )
        if self.can_bus_norm:
            self.can_bus_mlp.add_module('norm', nn.LayerNorm(self.embed_dims))
    
    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        normal_(self.level_embeds)
        normal_(self.cams_embeds)
        xavier_init(self.reference_points, distribution='uniform', bias=0.)
        xavier_init(self.can_bus_mlp, distribution='uniform', bias=0.)
    
    @auto_fp16(apply_to=('mlvl_feats', 'bev_queries', 'prev_bev', 'bev_pos'))
    def get_bev_features(
            self,
            mlvl_feats,
            bev_queries,
            bev_h,
            bev_w,
            grid_length=[0.512, 0.512],
            bev_pos=None,
            prev_bev=None,
            **kwargs):
        """Obtain BEV features.
        
        This function:
        1. Processes can_bus signals (rotation and translation)
        2. Rotates prev_bev if needed
        3. Adds camera and level embeddings
        4. Calls encoder to get BEV features
        
        Args:
            mlvl_feats: List of [B, num_cam, C, H_l, W_l] multi-scale features
            bev_queries: [bev_h*bev_w, embed_dims] BEV query embeddings
            bev_h, bev_w: BEV spatial dimensions
            grid_length: [grid_length_y, grid_length_x] grid size in meters
            bev_pos: [B, embed_dims, bev_h, bev_w] positional encoding
            prev_bev: [num_query, B, embed_dims] or [B, num_query, embed_dims] previous BEV
            
        Returns:
            bev_embed: [B, bev_h*bev_w, embed_dims] BEV features
        """
        bs = mlvl_feats[0].size(0)
        bev_queries = bev_queries.unsqueeze(1).repeat(1, bs, 1)  # [bev_h*bev_w, B, embed_dims]
        bev_pos = bev_pos.flatten(2).permute(2, 0, 1)  # [bev_h*bev_w, B, embed_dims]
        
        # Obtain rotation angle and shift with ego motion
        delta_x = np.array([each['can_bus'][0] for each in kwargs['img_metas']])
        delta_y = np.array([each['can_bus'][1] for each in kwargs['img_metas']])
        ego_angle = np.array([each['can_bus'][-2] / np.pi * 180 for each in kwargs['img_metas']])
        grid_length_y = grid_length[0]
        grid_length_x = grid_length[1]
        translation_length = np.sqrt(delta_x ** 2 + delta_y ** 2)
        translation_angle = np.arctan2(delta_y, delta_x) / np.pi * 180
        bev_angle = ego_angle - translation_angle
        shift_y = translation_length * np.cos(bev_angle / 180 * np.pi) / grid_length_y / bev_h
        shift_x = translation_length * np.sin(bev_angle / 180 * np.pi) / grid_length_x / bev_w
        shift_y = shift_y * self.use_shift
        shift_x = shift_x * self.use_shift
        shift = bev_queries.new_tensor([shift_x, shift_y]).permute(1, 0)  # [B, 2]
        
        # Rotate prev_bev if needed
        if prev_bev is not None:
            if prev_bev.shape[1] == bev_h * bev_w:
                prev_bev = prev_bev.permute(1, 0, 2)  # [bev_h*bev_w, B, embed_dims]
            if self.rotate_prev_bev:
                for i in range(bs):
                    rotation_angle = kwargs['img_metas'][i]['can_bus'][-1]
                    tmp_prev_bev = prev_bev[:, i].reshape(bev_h, bev_w, -1).permute(2, 0, 1)
                    tmp_prev_bev = rotate(tmp_prev_bev, rotation_angle, center=self.rotate_center)
                    tmp_prev_bev = tmp_prev_bev.permute(1, 2, 0).reshape(bev_h * bev_w, 1, -1)
                    prev_bev[:, i] = tmp_prev_bev[:, 0]
        
        # Add can bus signals
        can_bus = bev_queries.new_tensor([each['can_bus'] for each in kwargs['img_metas']])
        can_bus = self.can_bus_mlp(can_bus)[None, :, :]  # [1, B, embed_dims]
        bev_queries = bev_queries + can_bus * self.use_can_bus
        
        # Process multi-scale features: add camera and level embeddings
        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(mlvl_feats):
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            feat = feat.flatten(3).permute(1, 0, 3, 2)  # [num_cam, B, H*W, C]
            if self.use_cams_embeds:
                feat = feat + self.cams_embeds[:, None, None, :].to(feat.dtype)
            feat = feat + self.level_embeds[None, None, lvl:lvl + 1, :].to(feat.dtype)
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)
        
        feat_flatten = torch.cat(feat_flatten, 2)  # [num_cam, B, sum(H*W), C]
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=bev_pos.device)
        level_start_index = torch.cat((
            spatial_shapes.new_zeros((1,)),
            spatial_shapes.prod(1).cumsum(0)[:-1]))
        
        feat_flatten = feat_flatten.permute(0, 2, 1, 3)  # [num_cam, sum(H*W), B, embed_dims]
        
        # Call encoder
        bev_embed = self.encoder(
            bev_queries,  # [bev_h*bev_w, B, embed_dims]
            feat_flatten,  # [num_cam, sum(H*W), B, embed_dims]
            feat_flatten,  # same as key
            bev_h=bev_h,
            bev_w=bev_w,
            bev_pos=bev_pos,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            prev_bev=prev_bev,
            shift=shift,
            **kwargs
        )
        
        # Convert back to batch_first format: [B, bev_h*bev_w, embed_dims]
        if bev_embed.dim() == 3:
            bev_embed = bev_embed.permute(1, 0, 2)
        
        return bev_embed
    
    @auto_fp16(apply_to=('mlvl_feats', 'bev_queries', 'object_query_embed', 'prev_bev', 'bev_pos'))
    def forward(self,
                mlvl_feats,
                bev_queries,
                object_query_embed,
                bev_h,
                bev_w,
                grid_length=[0.512, 0.512],
                bev_pos=None,
                reg_branches=None,
                cls_branches=None,
                prev_bev=None,
                **kwargs):
        """Forward function for PerceptionTransformer.
        
        Args:
            mlvl_feats: List of [B, num_cam, C, H_l, W_l] multi-scale features
            bev_queries: [bev_h*bev_w, embed_dims] BEV query embeddings
            object_query_embed: [num_query, embed_dims*2] object query embeddings
            bev_h, bev_w: BEV spatial dimensions
            grid_length: [grid_length_y, grid_length_x] grid size in meters
            bev_pos: [B, embed_dims, bev_h, bev_w] positional encoding
            reg_branches: Regression heads for decoder (if with_box_refine)
            cls_branches: Classification heads for decoder (if as_two_stage)
            prev_bev: Previous BEV features
            
        Returns:
            tuple containing:
            - bev_embed: [B, bev_h*bev_w, embed_dims] BEV features
            - inter_states: [num_dec_layers, B, num_query, embed_dims] decoder outputs
            - init_reference_out: [B, num_query, 3] initial reference points
            - inter_references_out: [num_dec_layers, B, num_query, 3] reference points
        """
        # Get BEV features
        bev_embed = self.get_bev_features(
            mlvl_feats,
            bev_queries,
            bev_h,
            bev_w,
            grid_length=grid_length,
            bev_pos=bev_pos,
            prev_bev=prev_bev,
            **kwargs)  # [B, bev_h*bev_w, embed_dims]
        
        if self.decoder is None:
            # If no decoder, just return BEV features
            return bev_embed, None, None, None
        
        # Prepare object queries
        bs = mlvl_feats[0].size(0)
        query_pos, query = torch.split(object_query_embed, self.embed_dims, dim=1)
        query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)  # [B, num_query, embed_dims]
        query = query.unsqueeze(0).expand(bs, -1, -1)  # [B, num_query, embed_dims]
        
        # Generate reference points
        reference_points = self.reference_points(query_pos)  # [B, num_query, 3]
        reference_points = reference_points.sigmoid()
        init_reference_out = reference_points
        
        # Convert to sequence-first format for decoder
        query = query.permute(1, 0, 2)  # [num_query, B, embed_dims]
        query_pos = query_pos.permute(1, 0, 2)  # [num_query, B, embed_dims]
        bev_embed = bev_embed.permute(1, 0, 2)  # [bev_h*bev_w, B, embed_dims]
        
        # Call decoder
        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=bev_embed,
            query_pos=query_pos,
            reference_points=reference_points,
            reg_branches=reg_branches,
            cls_branches=cls_branches,
            spatial_shapes=torch.tensor([[bev_h, bev_w]], device=query.device, dtype=torch.long),
            level_start_index=torch.tensor([0], device=query.device, dtype=torch.long),
            **kwargs)
        
        inter_references_out = inter_references
        
        return bev_embed, inter_states, init_reference_out, inter_references_out
