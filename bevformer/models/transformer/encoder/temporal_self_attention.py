"""Temporal Self-Attention for BEVFormer."""
from typing import Optional
import torch
import torch.nn as nn
import math
import warnings

try:
    from mmcv.cnn import xavier_init, constant_init
    from mmcv.runner.base_module import BaseModule
except Exception:
    BaseModule = nn.Module
    def xavier_init(module, *args, **kwargs): pass
    def constant_init(module, *args, **kwargs): pass

from mmcv.cnn.bricks.registry import ATTENTION

try:
    from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch
    _has_mmcv_deform_attn = True
except Exception:
    _has_mmcv_deform_attn = False
    multi_scale_deformable_attn_pytorch = None

@ATTENTION.register_module()
class TemporalSelfAttention(BaseModule):
    """An attention module used in BEVFormer based on Deformable-Detr.
    
    `Deformable DETR: Deformable Transformers for End-to-End Object Detection.
    <https://arxiv.org/pdf/2010.04159.pdf>`_.
    
    This module fuses current BEV features with previous BEV features using
    deformable attention. It concatenates prev_bev and current query to predict
    sampling offsets and attention weights, then aggregates features from both
    temporal frames and fuses them by averaging.
    
    Args:
        embed_dims (int): The embedding dimension of Attention. Default: 256.
        num_heads (int): Parallel attention heads. Default: 8.
        num_levels (int): The number of feature map used in Attention. Default: 1.
        num_points (int): The number of sampling points for each query in each head.
            Default: 4.
        num_bev_queue (int): The length of BEV queue. In this version, we only use
            one history BEV and one current BEV, so the length is 2. Default: 2.
        im2col_step (int): The step used in image_to_column. Default: 64.
        dropout (float): A Dropout layer on `inp_identity`. Default: 0.1.
    """
    def __init__(self, 
                    embed_dims: int = 256, 
                    num_levels: int = 1,
                    num_points: int = 4,
                    num_heads: int = 8,
                    num_bev_queue: int = 2,
                    im2col_step: int = 64,
                    dropout: float = 0.1,
                    batch_first: bool = True,  # Accept but ignore (for compatibility with build_attention)
                    **kwargs) -> None:
        super().__init__(None)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                            f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        if not (dim_per_head & (dim_per_head - 1) == 0 and dim_per_head > 0):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')
        
        self.dropout = nn.Dropout(dropout)
        self.fp16_enabled = False
        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.num_bev_queue = num_bev_queue
        
        self.sampling_offsets = nn.Linear(
            embed_dims * num_bev_queue, 
            num_bev_queue * num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(
            embed_dims * num_bev_queue,
            num_bev_queue * num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.init_weights()
    
    def init_weights(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.sampling_offsets, 0.)
        # Generate circular pattern: [num_heads] -> [num_heads, 2] (cos, sin)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(
            self.num_heads, 1, 1, 2).repeat(1, self.num_levels * self.num_bev_queue, self.num_points, 1)
        # Scale by point index: point i gets scale (i+1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1
        self.sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
    
    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                **kwargs):
        """Forward Function of MultiScaleDeformAttention."""
        if value is None:
            bs, len_bev, c = query.shape
            value = torch.stack([query, query], 1).reshape(bs*2, len_bev, c)
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        
        bs, num_query, embed_dims = query.shape
        _, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value
        assert self.num_bev_queue == 2
        
        # Concatenate prev_bev and current query for offset/weight prediction
        # [bs, num_query, embed_dims] -> [bs, num_query, embed_dims*num_bev_queue]
        query = torch.cat([value[:bs], query], -1)
        # Project value: [bs*num_bev_queue, num_value, embed_dims]
        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        # Reshape for multi-head: [bs*num_bev_queue, num_value, num_heads, dim_per_head]
        value = value.reshape(bs*self.num_bev_queue, num_value, self.num_heads, -1)
        
        # Compute sampling offsets: [bs, num_query, num_heads, num_bev_queue, num_levels, num_points, 2]
        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_bev_queue, self.num_levels, self.num_points, 2)
        # Compute attention weights: [bs, num_query, num_heads, num_bev_queue, num_levels*num_points]
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_bev_queue, self.num_levels * self.num_points)
        attention_weights = attention_weights.softmax(-1).view(
            bs, num_query, self.num_heads, self.num_bev_queue, self.num_levels, self.num_points)
        
        # Permute for processing: [bs*num_bev_queue, num_query, num_heads, num_levels, num_points]
        attention_weights = attention_weights.permute(0, 3, 1, 2, 4, 5)\
            .reshape(bs*self.num_bev_queue, num_query, self.num_heads, self.num_levels, self.num_points).contiguous()
        sampling_offsets = sampling_offsets.permute(0, 3, 1, 2, 4, 5, 6)\
            .reshape(bs*self.num_bev_queue, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        
        # Compute sampling locations
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                + sampling_offsets / self.num_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')
        
        # Apply deformable attention
        if _has_mmcv_deform_attn and multi_scale_deformable_attn_pytorch is not None:
            try:
                output = multi_scale_deformable_attn_pytorch(
                    value, spatial_shapes, sampling_locations, attention_weights)
            except RuntimeError as e:
                # Fallback to PyTorch implementation if CUDA kernel fails (e.g., sm_120 compatibility)
                if 'CUDA' in str(e) or 'kernel' in str(e).lower():
                    output = self._deformable_attn_pytorch(
                        value, spatial_shapes, sampling_locations, attention_weights)
                else:
                    raise
        else:
            # Fallback to PyTorch implementation
            output = self._deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights)
        
        # Fuse history value and current value
        # [bs*num_bev_queue, num_query, embed_dims] -> [num_query, embed_dims, bs*num_bev_queue]
        output = output.permute(1, 2, 0)
        # [num_query, embed_dims, bs*num_bev_queue] -> [num_query, embed_dims, bs, num_bev_queue]
        output = output.view(num_query, embed_dims, bs, self.num_bev_queue).mean(-1)
        # [num_query, embed_dims, bs] -> [bs, num_query, embed_dims]
        output = output.permute(2, 0, 1)
        output = self.output_proj(output)
        return self.dropout(output) + identity
    
    def _deformable_attn_pytorch(self, value, spatial_shapes, sampling_locations, attention_weights):
        """PyTorch fallback implementation using grid_sample for bilinear sampling.
        
        This is used when mmcv's CUDA implementation is not available or fails due to
        CUDA compatibility issues (e.g., sm_120 not supported).
        
        Args:
            value: [bs*num_bev_queue, num_value, num_heads, dim_per_head]
            spatial_shapes: [num_levels, 2]
            sampling_locations: [bs*num_bev_queue, num_query, num_heads, num_levels, num_points, 2]
            attention_weights: [bs*num_bev_queue, num_query, num_heads, num_levels, num_points]
            
        Returns:
            output: [bs*num_bev_queue, num_query, num_heads, dim_per_head]
        """
        bs_queue, num_value, num_heads, dim_per_head = value.shape
        num_query = sampling_locations.shape[1]
        
        # Split by level: [bs_queue, H_l*W_l, num_heads, dim_per_head]
        value_list = value.split([int(H * W) for H, W in spatial_shapes], dim=1)
        output = value.new_zeros(bs_queue, num_query, num_heads, dim_per_head)
        
        for level_id, (H, W) in enumerate(spatial_shapes):
            # Reshape to spatial: [bs_queue, H*W, num_heads, dim_per_head] -> [bs_queue*num_heads, dim_per_head, H, W]
            value_lvl = value_list[level_id].view(bs_queue, H, W, num_heads, dim_per_head)
            value_lvl = value_lvl.permute(0, 3, 4, 1, 2).contiguous()
            value_lvl = value_lvl.view(bs_queue * num_heads, dim_per_head, H, W)
            
            # Extract locations: [bs_queue, num_query, num_heads, num_points, 2]
            sampling_loc_lvl = sampling_locations[:, :, :, level_id, :, :]
            # Convert [0, 1] -> [-1, 1] and reshape: [bs_queue*num_heads, num_query, num_points, 2]
            sampling_loc_lvl = (sampling_loc_lvl * 2.0 - 1.0).permute(0, 2, 1, 3, 4).contiguous()
            sampling_loc_lvl = sampling_loc_lvl.view(bs_queue * num_heads, num_query, self.num_points, 2)
            
            # Sample: [bs_queue*num_heads, dim_per_head, H, W] -> [bs_queue*num_heads, dim_per_head, num_query, num_points]
            sampled_value = F.grid_sample(
                value_lvl, sampling_loc_lvl, mode='bilinear', padding_mode='zeros', align_corners=True)
            
            # Reshape: [bs_queue, num_query, num_heads, dim_per_head, num_points]
            sampled_value = sampled_value.view(bs_queue, num_heads, dim_per_head, num_query, self.num_points)
            sampled_value = sampled_value.permute(0, 3, 1, 2, 4)
            
            # Weighted sum: [bs_queue, num_query, num_heads, dim_per_head]
            attn_weight_lvl = attention_weights[:, :, :, level_id, :].unsqueeze(3)
            output_lvl = (sampled_value * attn_weight_lvl).sum(dim=4)
            output = output + output_lvl
        
        return output
