"""Temporal Self-Attention for BEVFormer, aligned with original implementation."""
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import warnings

try:
    from mmcv.cnn import xavier_init, constant_init
    from mmcv.runner.base_module import BaseModule
except Exception:
    BaseModule = nn.Module
    def xavier_init(module, *args, **kwargs): pass
    def constant_init(module, *args, **kwargs): pass

try:
    from mmdet.models.utils.builder import ATTENTION
except Exception:
    try:
        from mmdet.models.builder import ATTENTION
    except Exception:
        ATTENTION = None

from .ms_deform_attn_3d import MSDeformableAttention3D

@ATTENTION.register_module()
class TemporalSelfAttention(BaseModule):
    """Temporal self-attention using deformable attention.
    
    This module fuses current BEV with previous BEV using deformable attention.
    The interface is compatible with the original BEVFormer implementation.
    
    Args:
        embed_dims: feature dimension
        num_levels: number of levels (should be 2 for temporal: [prev_bev, current_bev])
        num_points: number of sampling points per head
        num_heads: number of attention heads
        num_bev_queue: number of BEV frames in queue (default 2)
        batch_first: whether batch dimension is first
    """
    def __init__(self, 
                    embed_dims: int = 256, 
                    num_levels: int = 2,  # Two temporal frames: [prev_bev, current_bev]
                    num_points: int = 8,
                    num_heads: int = 8,
                    num_bev_queue: int = 2,
                    im2col_step: int = 64,
                    dropout: float = 0.1,
                    batch_first: bool = True,
                    norm_cfg: Optional[dict] = None,
                    init_cfg: Optional[dict] = None) -> None:
        super().__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                            f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first
        self.fp16_enabled = False       
        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.num_bev_queue = num_bev_queue
        
        # Use MSDeformableAttention3D for temporal fusion
        # Note: We use a simplified implementation that matches the interface
        self.deform_attn = MSDeformableAttention3D(
            embed_dims=embed_dims,
            num_levels=num_levels,
            num_points=num_points,
            num_heads=num_heads
        )
        
    
    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                reference_points=None,
                spatial_shapes=None,
                **kwargs):
        """Forward function for temporal self-attention.
        
        This interface is compatible with the original BEVFormer implementation.
        
        Args:
            query: [bs, num_query, embed_dims] or [num_query, bs, embed_dims] current BEV
            key: [bs, num_query, embed_dims] previous BEV (ignored, use value instead)
            value: [bs*num_bev_queue, num_query, embed_dims] stacked BEV features
            identity: [bs, num_query, embed_dims] residual connection
            query_pos: [bs, num_query, embed_dims] positional encoding
            reference_points: [bs, num_query, num_levels, 2] 2D reference points
            spatial_shapes: [num_levels, 2] spatial shapes
            level_start_index: [num_levels] level start indices
            
        Returns:
            output: [bs, num_query, embed_dims] fused BEV features
        """
        if value is None:
            # If value is None, create from query (no prev_bev case)
            assert self.batch_first
            bs, len_bev, c = query.shape
            value = torch.stack([query, query], 1).reshape(bs*2, len_bev, c)
        
        if identity is None:
            identity = query
        
        if query_pos is not None:
            query = query + query_pos
        
        if not self.batch_first:
            # Convert to batch_first format
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)
        
        bs, num_query, embed_dims = query.shape
        _, num_value, _ = value.shape
        
        # Verify spatial shapes match
        if spatial_shapes is not None:
            assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value, \
                f"spatial_shapes mismatch: {(spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum()} != {num_value}"
        
        assert self.num_bev_queue == 2, "num_bev_queue must be 2 for temporal attention"
        
        # Reshape value to [bs, num_bev_queue, num_query, embed_dims]
        value_reshaped = value.view(bs, self.num_bev_queue, num_query, embed_dims)
        
        # Convert to feature maps: [prev_bev, current_bev]
        # Extract spatial dimensions from reference_points or use default
        if reference_points is not None:
            # reference_points: [bs, num_query, num_levels, 2]
            # Infer H, W from spatial_shapes
            if spatial_shapes is not None:
                bev_h, bev_w = int(spatial_shapes[0, 0].item()), int(spatial_shapes[0, 1].item())
            else:
                # Default: assume square BEV
                bev_h = bev_w = int(math.sqrt(num_query))
        else:
            # Default: assume square BEV
            bev_h = bev_w = int(math.sqrt(num_query))
        
        # Reshape BEV features to spatial format
        prev_bev_feat = value_reshaped[:, 0].permute(0, 2, 1).reshape(bs, embed_dims, bev_h, bev_w)
        current_bev_feat = value_reshaped[:, 1].permute(0, 2, 1).reshape(bs, embed_dims, bev_h, bev_w)
        
        # Prepare reference points for deformable attention
        if reference_points is not None:
            # reference_points: [bs, num_query, num_levels, 2]
            # Convert from [0, 1] to [-1, 1] for grid_sample
            ref_points = reference_points * 2.0 - 1.0  # [bs, num_query, num_levels, 2]
        else:
            # Generate default reference points
            xs = torch.linspace(-1.0, 1.0, bev_w, device=query.device)
            ys = torch.linspace(-1.0, 1.0, bev_h, device=query.device)
            yy, xx = torch.meshgrid(ys, xs, indexing='ij')
            ref_points_xy = torch.stack([xx, yy], dim=-1).reshape(1, num_query, 2)
            ref_points = ref_points_xy.unsqueeze(2).repeat(bs, 1, self.num_levels, 1)
        
        # Apply deformable attention
        feats = [prev_bev_feat, current_bev_feat]  # List of 2 [B, C, H, W]
        out = self.deform_attn(query, feats, ref_points)
        
        # Residual connection and dropout
        output = self.dropout(out) + identity
        
        return output
