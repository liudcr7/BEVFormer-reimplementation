"""Spatial Cross-Attention for BEVFormer, aligned with original implementation."""
from typing import List, Tuple, Optional
import torch
import torch.nn as nn
import warnings

try:
    from mmcv.cnn import xavier_init, constant_init
    from mmcv.runner.base_module import BaseModule
    from mmcv.runner import force_fp32
    from mmcv.cnn.bricks.transformer import build_attention
except Exception:
    BaseModule = nn.Module
    def xavier_init(module, *args, **kwargs): pass
    def constant_init(module, *args, **kwargs): pass
    def force_fp32(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    def build_attention(cfg): return None

try:
    from mmdet.models.utils.builder import ATTENTION
except Exception:
    try:
        from mmdet.models.builder import ATTENTION
    except Exception:
        ATTENTION = None

from .ms_deform_attn_3d import MSDeformableAttention3D

@ATTENTION.register_module()
class SpatialCrossAttention(BaseModule):
    """Spatial cross-attention across multiple camera views.
    
    This module aggregates features from multi-view images to BEV queries
    using deformable attention. The interface is compatible with the original
    BEVFormer implementation.
    
    Args:
        embed_dims: feature dimension
        num_cams: number of cameras
        pc_range: point cloud range
        dropout: dropout rate
        deformable_attention: config for deformable attention
    """
    def __init__(self,
                    embed_dims: int = 256,
                    num_cams: int = 6,
                    pc_range: Optional[List[float]] = None,
                    dropout: float = 0.1,
                    init_cfg: Optional[dict] = None,
                    batch_first: bool = False,
                    deformable_attention: Optional[dict] = None,
                    **kwargs):
        super(SpatialCrossAttention, self).__init__(init_cfg)
        
        self.init_cfg = init_cfg
        self.dropout = nn.Dropout(dropout)
        self.pc_range = pc_range
        self.fp16_enabled = False
        self.embed_dims = embed_dims
        self.num_cams = num_cams
        self.batch_first = batch_first
        
        # Build deformable attention
        if deformable_attention is not None:
            self.deformable_attention = build_attention(deformable_attention)
        else:
            # Default: use MSDeformableAttention3D
            self.deformable_attention = MSDeformableAttention3D(
                embed_dims=embed_dims,
                num_levels=4,
                num_points=8,
                num_heads=8
            )
        
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.init_weight()
    
    def init_weight(self):
        """Default initialization for Parameters of Module."""
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
    
    @force_fp32(apply_to=('query', 'key', 'value', 'query_pos', 'reference_points_cam'))
    def forward(self,
                query,
                key,
                value,
                residual=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                reference_points_cam=None,
                bev_mask=None,
                level_start_index=None,
                flag='encoder',
                **kwargs):
        """Forward function for spatial cross-attention.
        
        This interface is compatible with the original BEVFormer implementation.
        
        Args:
            query: [bs, num_query, embed_dims] BEV queries
            key: [num_cam, num_value, bs, embed_dims] multi-camera features
            value: same as key
            residual: [bs, num_query, embed_dims] residual connection
            query_pos: [bs, num_query, embed_dims] positional encoding
            reference_points: [bs, num_query, D, 3] 3D reference points
            reference_points_cam: [num_cam, bs, num_query, D, 2] projected points
            bev_mask: [num_cam, bs, num_query, D] visibility mask
            spatial_shapes: [num_levels, 2] spatial shapes
            level_start_index: [num_levels] level start indices
            
        Returns:
            output: [bs, num_query, embed_dims] aggregated BEV features
        """
        if key is None:
            key = query
        if value is None:
            value = key
        
        if residual is None:
            inp_residual = query
            slots = torch.zeros_like(query)
        else:
            inp_residual = residual
            slots = torch.zeros_like(query)
        
        if query_pos is not None:
            query = query + query_pos
        
        bs, num_query, _ = query.size()
        
        if reference_points_cam is None or bev_mask is None:
            # Fallback: use simplified implementation
            # This should not happen in normal flow, but handle gracefully
            warnings.warn("reference_points_cam or bev_mask is None, using fallback")
            return self.dropout(slots) + inp_residual
        
        D = reference_points_cam.size(3)
        
        # Find valid queries for each camera (queries visible in that camera)
        indexes = []
        for i, mask_per_img in enumerate(bev_mask):
            # mask_per_img: [bs, num_query, D]
            # Find queries that are visible in at least one level
            index_query_per_img = mask_per_img[0].sum(-1).nonzero().squeeze(-1)
            indexes.append(index_query_per_img)
        
        max_len = max([len(each) for each in indexes]) if indexes else num_query
        
        # Rebatch queries and reference points for efficient processing
        # Each camera only interacts with its corresponding BEV queries
        queries_rebatch = query.new_zeros([bs, self.num_cams, max_len, self.embed_dims])
        reference_points_rebatch = reference_points_cam.new_zeros(
            [bs, self.num_cams, max_len, D, 2])
        
        for j in range(bs):
            for i, reference_points_per_img in enumerate(reference_points_cam):
                index_query_per_img = indexes[i] if i < len(indexes) else torch.arange(num_query, device=query.device)
                if len(index_query_per_img) > 0:
                    queries_rebatch[j, i, :len(index_query_per_img)] = query[j, index_query_per_img]
                    reference_points_rebatch[j, i, :len(index_query_per_img)] = reference_points_per_img[j, index_query_per_img]
        
        # Reshape key and value: [num_cam, num_value, bs, embed_dims] -> [bs*num_cams, num_value, embed_dims]
        num_cams, l, bs_k, embed_dims = key.shape
        assert bs_k == bs, f"batch size mismatch: {bs_k} != {bs}"
        
        key = key.permute(2, 0, 1, 3).reshape(bs * self.num_cams, l, self.embed_dims)
        value = value.permute(2, 0, 1, 3).reshape(bs * self.num_cams, l, self.embed_dims)
        
        # Apply deformable attention per camera
        queries = self.deformable_attention(
            query=queries_rebatch.view(bs*self.num_cams, max_len, self.embed_dims),
            feats=[key, value],  # Simplified: use key and value as feature maps
            reference_points=reference_points_rebatch.view(bs*self.num_cams, max_len, D, 2)
        ).view(bs, self.num_cams, max_len, self.embed_dims)
        
        # Aggregate results from all cameras
        for j in range(bs):
            for i, index_query_per_img in enumerate(indexes):
                if i < len(indexes) and len(index_query_per_img) > 0:
                    slots[j, index_query_per_img] += queries[j, i, :len(index_query_per_img)]
        
        # Normalize by number of cameras that see each query
        count = bev_mask.sum(-1) > 0  # [num_cam, bs, num_query]
        count = count.permute(1, 2, 0).sum(-1)  # [bs, num_query]
        count = torch.clamp(count, min=1.0)
        slots = slots / count[..., None]
        
        # Output projection
        slots = self.output_proj(slots)
        
        return self.dropout(slots) + inp_residual
