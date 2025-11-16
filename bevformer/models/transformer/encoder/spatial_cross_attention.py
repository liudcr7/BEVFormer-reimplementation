"""Spatial Cross-Attention for BEVFormer."""
from typing import List, Optional
import torch
import torch.nn as nn
import warnings

try:
    from mmcv.cnn import xavier_init
    from mmcv.runner.base_module import BaseModule
    from mmcv.runner import force_fp32
    from mmcv.cnn.bricks.transformer import build_attention
except Exception:
    BaseModule = nn.Module
    def xavier_init(module, *args, **kwargs): pass
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

@ATTENTION.register_module()
class SpatialCrossAttention(BaseModule):
    """An attention module used in BEVFormer.
    
    This module aggregates features from multi-view images to BEV queries
    using deformable attention. Each camera only interacts with its corresponding
    BEV queries to save GPU memory.
    
    Args:
        embed_dims (int): The embedding dimension of Attention. Default: 256.
        num_cams (int): The number of cameras. Default: 6.
        pc_range (List[float]): Point cloud range. Default: None.
        dropout (float): A Dropout layer on `inp_residual`. Default: 0.1.
        deformable_attention (dict): The config for the deformable attention used in SCA.
    
    Returns:
        Tensor: forwarded results with shape [bs, num_query, embed_dims].
    """
    def __init__(self,
                    embed_dims: int = 256,
                    num_cams: int = 6,
                    pc_range: Optional[List[float]] = None,
                    dropout: float = 0.1,
                    deformable_attention: Optional[dict] = None,
                    **kwargs):
        super().__init__(None)
        self.dropout = nn.Dropout(dropout)
        self.pc_range = pc_range
        self.fp16_enabled = False
        self.embed_dims = embed_dims
        self.num_cams = num_cams
        self.deformable_attention = build_attention(deformable_attention)
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
                **kwargs):
        """Forward Function of Detr3DCrossAtten."""
        if key is None:
            key = query
        if value is None:
            value = key
        
        if residual is None:
            inp_residual = query  # [bs, num_query, embed_dims]
            slots = torch.zeros_like(query)  # [bs, num_query, embed_dims]
        else:
            inp_residual = residual  # [bs, num_query, embed_dims]
            slots = torch.zeros_like(query)  # [bs, num_query, embed_dims]
        if query_pos is not None:
            query = query + query_pos  # [bs, num_query, embed_dims]
        
        bs, num_query, _ = query.size()  # Extract dimensions
        D = reference_points_cam.size(3)  # num_points_in_pillar (typically 4)
        
        # Find valid queries for each camera
        # For each camera, find which BEV queries have at least one visible point
        indexes = []
        for i, mask_per_img in enumerate(bev_mask):
            # mask_per_img: [bs, num_query, D] -> sum(-1): [bs, num_query] -> [0]: [num_query]
            # nonzero().squeeze(-1): [num_visible_queries]
            index_query_per_img = mask_per_img[0].sum(-1).nonzero().squeeze(-1)
            indexes.append(index_query_per_img)
        max_len = max([len(each) for each in indexes]) if indexes else num_query
        
        # Rebatch queries and reference points for efficient processing
        # Create fixed-size tensors: [bs, num_cams, max_len, embed_dims]
        queries_rebatch = query.new_zeros([bs, self.num_cams, max_len, self.embed_dims])
        # [bs, num_cams, max_len, D, 2]
        reference_points_rebatch = reference_points_cam.new_zeros(
            [bs, self.num_cams, max_len, D, 2])
        
        # Fill rebatched tensors with visible queries and their reference points
        for j in range(bs):
            for i, reference_points_per_img in enumerate(reference_points_cam):
                index_query_per_img = indexes[i]
                # Copy visible queries: [num_visible, embed_dims] -> [max_len, embed_dims] (padded)
                queries_rebatch[j, i, :len(index_query_per_img)] = query[j, index_query_per_img]
                # Copy reference points: [num_visible, D, 2] -> [max_len, D, 2] (padded)
                reference_points_rebatch[j, i, :len(index_query_per_img)] = reference_points_per_img[j, index_query_per_img]
        
        # Reshape key and value for batch processing
        # key/value: [num_cam, num_value, bs, embed_dims] -> [bs*num_cams, num_value, embed_dims]
        num_cams, l, bs_k, embed_dims = key.shape
        key = key.permute(2, 0, 1, 3).reshape(bs * self.num_cams, l, self.embed_dims)
        value = value.permute(2, 0, 1, 3).reshape(bs * self.num_cams, l, self.embed_dims)
        
        # Apply deformable attention per camera
        # queries_rebatch: [bs, num_cams, max_len, embed_dims] -> [bs*num_cams, max_len, embed_dims]
        # reference_points_rebatch: [bs, num_cams, max_len, D, 2] -> [bs*num_cams, max_len, D, 2]
        queries = self.deformable_attention(
            query=queries_rebatch.view(bs*self.num_cams, max_len, self.embed_dims),
            key=key,  # [bs*num_cams, num_value, embed_dims]
            value=value,  # [bs*num_cams, num_value, embed_dims]
            reference_points=reference_points_rebatch.view(bs*self.num_cams, max_len, D, 2),
            spatial_shapes=spatial_shapes,  # [num_levels, 2]
            level_start_index=level_start_index  # [num_levels]
        ).view(bs, self.num_cams, max_len, self.embed_dims)  # [bs*num_cams, max_len, embed_dims] -> [bs, num_cams, max_len, embed_dims]
        
        # Aggregate results from all cameras back to original query positions
        # For each camera, add its output to the corresponding query slots
        for j in range(bs):
            for i, index_query_per_img in enumerate(indexes):
                # queries[j, i, :len(index_query_per_img)]: [num_visible, embed_dims]
                # slots[j, index_query_per_img]: [num_visible, embed_dims]
                slots[j, index_query_per_img] += queries[j, i, :len(index_query_per_img)]
        
        # Normalize by number of cameras that see each query
        # bev_mask.sum(-1): [num_cam, bs, num_query, D] -> [num_cam, bs, num_query]
        # > 0: [num_cam, bs, num_query] (bool) - True if visible in at least one level
        count = bev_mask.sum(-1) > 0
        # permute(1, 2, 0): [num_cam, bs, num_query] -> [bs, num_query, num_cam]
        # sum(-1): [bs, num_query, num_cam] -> [bs, num_query] - count cameras per query
        count = count.permute(1, 2, 0).sum(-1)
        count = torch.clamp(count, min=1.0)  # Avoid division by zero
        # slots: [bs, num_query, embed_dims] / count[..., None]: [bs, num_query, 1] -> [bs, num_query, embed_dims]
        slots = slots / count[..., None]
        # Output projection: [bs, num_query, embed_dims] -> [bs, num_query, embed_dims]
        slots = self.output_proj(slots)
        
        return self.dropout(slots) + inp_residual  # [bs, num_query, embed_dims]
