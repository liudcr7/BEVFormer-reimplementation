"""Custom Multi-Scale Deformable Attention for decoder cross-attention."""
import warnings
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init, constant_init
from mmcv.runner.base_module import BaseModule

# Try to import mmcv's optimized implementation
# mmcv provides an optimized PyTorch implementation of multi-scale deformable attention
# that may be faster than our grid_sample-based implementation
try:
    from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch
    _has_mmcv_deform_attn = True
except Exception:
    # Fallback: mmcv not available, will use our PyTorch implementation
    _has_mmcv_deform_attn = False
    multi_scale_deformable_attn_pytorch = None

# Try to import ATTENTION registry for module registration
# This allows the module to be built from config files
try:
    from mmdet.models.utils.builder import ATTENTION
except Exception:
    try:
        from mmdet.models.builder import ATTENTION
    except Exception:
        ATTENTION = None


if ATTENTION is not None:
    @ATTENTION.register_module()
    class CustomMSDeformableAttention(BaseModule):
        """Multi-Scale Deformable Attention for decoder cross-attention.
        
        This attention module performs cross-attention between object queries and multi-scale
        BEV features using deformable sampling. Unlike standard attention that attends to all
        positions, deformable attention only samples from a small number of key positions
        determined by learnable offsets, making it more efficient.
        
        The attention process:
        1. For each object query, predict sampling offsets and attention weights
        2. Sample features from multi-scale BEV features at offset locations
        3. Aggregate sampled features using attention weights
        4. Project aggregated features to output dimension
        
        This implementation supports both mmcv's optimized version (when available) and
        a fallback PyTorch implementation using grid_sample for bilinear sampling.
        
        Args:
            embed_dims (int): Feature dimension of queries and values.
                Default: 256.
            num_heads (int): Number of parallel attention heads. Each head processes
                embed_dims // num_heads dimensions. Default: 8.
            num_levels (int): Number of feature pyramid levels to attend to.
                Typically 4 for FPN features. Default: 4.
            num_points (int): Number of sampling points per head per level.
                More points allow finer-grained attention but increase computation.
                Default: 4.
            dropout (float): Dropout probability applied to output.
                Default: 0.1.
        """
        
        def __init__(self,
                     embed_dims=256,
                     num_heads=8,
                     num_levels=4,
                     num_points=4,
                     dropout=0.1,
                     **kwargs):
            # Validate: embed_dims must be divisible by num_heads
            if embed_dims % num_heads != 0:
                raise ValueError(
                    f'embed_dims ({embed_dims}) must be divisible by num_heads ({num_heads})')
            
            dim_per_head = embed_dims // num_heads
            
            # Warn if dim_per_head is not power of 2
            if not (dim_per_head & (dim_per_head - 1) == 0 and dim_per_head > 0):
                warnings.warn(
                    f'dim_per_head ({dim_per_head}) should be a power of 2 '
                    'for better CUDA efficiency in deformable attention.')
            
            self.embed_dims = embed_dims
            self.num_heads = num_heads
            self.num_levels = num_levels
            self.num_points = num_points
            self.dim_per_head = dim_per_head
            self.fp16_enabled = False
            
            # Linear layers: [embed_dims] -> [num_heads * num_levels * num_points * 2]
            self.sampling_offsets = nn.Linear(
                embed_dims, num_heads * num_levels * num_points * 2)
            # Linear layers: [embed_dims] -> [num_heads * num_levels * num_points]
            self.attention_weights = nn.Linear(
                embed_dims, num_heads * num_levels * num_points)
            self.value_proj = nn.Linear(embed_dims, embed_dims)
            self.output_proj = nn.Linear(embed_dims, embed_dims)
            self.dropout = nn.Dropout(dropout)
            
            self.init_weights()
        
        def init_weights(self):
            """Initialize weights with circular pattern for sampling offsets.
            
            This method initializes the attention module weights:
            1. Sampling offsets: Initialized with circular pattern around reference points
               - Each head gets a different angle (evenly distributed around circle)
               - Points at different distances have different offset scales
               - This helps the model learn spatial relationships from the start
            2. Attention weights: Initialized to zero (uniform attention initially)
            3. Value and output projections: Initialized with Xavier uniform
            """
            constant_init(self.sampling_offsets, 0.)
            
            # Generate angles: [num_heads] -> [num_heads, 2] (cos, sin)
            thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
            grid_init = torch.stack([thetas.cos(), thetas.sin()], dim=-1)
            
            # Normalize and reshape: [num_heads, 2] -> [num_heads, num_levels, num_points, 2]
            grid_init = (grid_init / grid_init.abs().max(dim=-1, keepdim=True)[0]).view(
                self.num_heads, 1, 1, 2).repeat(1, self.num_levels, self.num_points, 1)
            
            # Scale by point index: point i gets scale (i+1)
            for i in range(self.num_points):
                grid_init[:, :, i, :] *= i + 1
            
            # Set bias: [num_heads * num_levels * num_points * 2]
            self.sampling_offsets.bias.data = grid_init.view(-1)
            
            constant_init(self.attention_weights, val=0., bias=0.)
            xavier_init(self.value_proj, distribution='uniform', bias=0.)
            xavier_init(self.output_proj, distribution='uniform', bias=0.)
        
        def _compute_sampling_locations(self, reference_points, sampling_offsets, spatial_shapes):
            """Compute actual sampling locations from reference points and offsets.
            
            This function combines reference points (initial locations) with learnable offsets
            to determine where to sample features from. The normalization depends on whether
            reference points are 2D coordinates or 4D boxes.
            
            Args:
                reference_points (torch.Tensor): Reference points for sampling.
                    - Shape: [bs, num_query, num_levels, 2] for 2D coordinates
                    - Shape: [bs, num_query, num_levels, 4] for boxes (x, y, w, h)
                    - Values are normalized to [0, 1] range
                sampling_offsets (torch.Tensor): Learnable offsets from query features.
                    - Shape: [bs, num_query, num_heads, num_levels, num_points, 2]
                    - Values are in pixel/feature space (not normalized)
                spatial_shapes (torch.Tensor): Spatial dimensions for each feature level.
                    - Shape: [num_levels, 2]
                    - Each row: [H_l, W_l] for level l
                    
            Returns:
                torch.Tensor: Final sampling locations in normalized coordinates [0, 1].
                    - Shape: [bs, num_query, num_heads, num_levels, num_points, 2]
                    - Format: (x, y) coordinates normalized to [0, 1] for grid_sample
            """
            if reference_points.shape[-1] == 2:
                # Normalize offsets by spatial dimensions: [num_levels, 2] -> (W, H)
                offset_normalizer = torch.stack(
                    [spatial_shapes[..., 1], spatial_shapes[..., 0]], dim=-1)
                # [bs, num_query, num_levels, 2] + [bs, num_query, num_heads, num_levels, num_points, 2] / [num_levels, 2]
                sampling_locations = reference_points[:, :, None, :, None, :] + \
                    sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            elif reference_points.shape[-1] == 4:
                # Scale offsets by box size
                sampling_locations = reference_points[:, :, None, :, None, :2] + \
                    sampling_offsets / self.num_points * reference_points[:, :, None, :, None, 2:] * 0.5
            else:
                raise ValueError(
                    f'reference_points last dim must be 2 or 4, got {reference_points.shape[-1]}')
            
            return sampling_locations
        
        def _deformable_attn_pytorch(self, value, spatial_shapes, sampling_locations, attention_weights):
            """PyTorch implementation using grid_sample for bilinear sampling.
            
            This is a fallback implementation when mmcv's optimized version is not available.
            It uses PyTorch's grid_sample for efficient bilinear interpolation, which provides
            good performance while being more portable than CUDA extensions.
            
            The implementation processes each feature level separately:
            1. Reshape value features to spatial format (H, W)
            2. Extract sampling locations for this level
            3. Sample features using bilinear interpolation
            4. Apply attention weights and aggregate
            5. Accumulate results across all levels
            
            Args:
                value (torch.Tensor): Projected value features from all levels concatenated.
                    - Shape: [bs, num_value, num_heads, dim_per_head]
                    - num_value = sum(H_l * W_l) for all levels l
                spatial_shapes (torch.Tensor): Spatial dimensions for each feature level.
                    - Shape: [num_levels, 2]
                    - Each row: [H_l, W_l] for level l
                sampling_locations (torch.Tensor): Normalized sampling locations.
                    - Shape: [bs, num_query, num_heads, num_levels, num_points, 2]
                    - Values in [0, 1] range (will be converted to [-1, 1] for grid_sample)
                attention_weights (torch.Tensor): Attention weights for each sampling point.
                    - Shape: [bs, num_query, num_heads, num_levels, num_points]
                    - Already softmaxed, sums to 1 across points for each level
                    
            Returns:
                torch.Tensor: Aggregated attention output.
                    - Shape: [bs, num_query, num_heads, dim_per_head]
                    - Features aggregated across all levels and sampling points
            """
            bs, num_value, num_heads, dim_per_head = value.shape
            num_query = sampling_locations.shape[1]
            
            # Split by level: [bs, H_l*W_l, num_heads, dim_per_head]
            value_list = value.split([int(H * W) for H, W in spatial_shapes], dim=1)
            output = value.new_zeros(bs, num_query, num_heads, dim_per_head)
            
            for level_id, (H, W) in enumerate(spatial_shapes):
                # Reshape to spatial: [bs, H*W, num_heads, dim_per_head] -> [bs*num_heads, dim_per_head, H, W]
                value_lvl = value_list[level_id].view(bs, H, W, num_heads, dim_per_head)
                value_lvl = value_lvl.permute(0, 3, 4, 1, 2).contiguous()
                value_lvl = value_lvl.view(bs * num_heads, dim_per_head, H, W)
                
                # Extract locations: [bs, num_query, num_heads, num_points, 2]
                sampling_loc_lvl = sampling_locations[:, :, :, level_id, :, :]
                # Convert [0, 1] -> [-1, 1] and reshape: [bs*num_heads, num_query, num_points, 2]
                sampling_loc_lvl = (sampling_loc_lvl * 2.0 - 1.0).permute(0, 2, 1, 3, 4).contiguous()
                sampling_loc_lvl = sampling_loc_lvl.view(bs * num_heads, num_query, num_points, 2)
                
                # Sample: [bs*num_heads, dim_per_head, H, W] -> [bs*num_heads, dim_per_head, num_query, num_points]
                sampled_value = F.grid_sample(
                    value_lvl, sampling_loc_lvl, mode='bilinear', padding_mode='zeros', align_corners=True)
                
                # Reshape: [bs, num_query, num_heads, dim_per_head, num_points]
                sampled_value = sampled_value.view(bs, num_heads, dim_per_head, num_query, num_points)
                sampled_value = sampled_value.permute(0, 3, 1, 2, 4)
                
                # Weighted sum: [bs, num_query, num_heads, dim_per_head]
                attn_weight_lvl = attention_weights[:, :, :, level_id, :].unsqueeze(3)
                output_lvl = (sampled_value * attn_weight_lvl).sum(dim=4)
                output = output + output_lvl
            
            return output
        
        def forward(self,
                    query,
                    value=None,
                    identity=None,
                    query_pos=None,
                    key_padding_mask=None,
                    reference_points=None,
                    spatial_shapes=None,
                    **kwargs):
            """Forward pass of multi-scale deformable attention.
            
            This function performs the complete attention computation:
            1. Processes query and value inputs (add positional encoding, convert format)
            2. Projects value features
            3. Computes sampling offsets and attention weights from queries
            4. Computes sampling locations from reference points and offsets
            5. Samples and aggregates features using deformable attention
            6. Projects output and applies residual connection
            
            Args:
                query (torch.Tensor): Object queries (what to attend to).
                    - Shape: [num_query, bs, embed_dims] (sequence-first format)
                    - num_query: number of object queries (typically 900)
                value (torch.Tensor, optional): BEV features to attend to.
                    - Shape: [num_key, bs, embed_dims] (sequence-first format)
                    - num_key: total number of BEV features (sum of H_l * W_l for all levels)
                    - If None, uses query as value. Default: None.
                identity (torch.Tensor, optional): Tensor for residual connection.
                    - Same shape as query
                    - If None, uses query. Default: None.
                query_pos (torch.Tensor, optional): Positional encoding for queries.
                    - Same shape as query
                    - Added to query before computing offsets/weights. Default: None.
                key_padding_mask (torch.Tensor, optional): Mask for invalid positions.
                    - Shape: [bs, num_key]
                    - True/1 for invalid positions, False/0 for valid
                    - Invalid positions are masked to zero. Default: None.
                reference_points (torch.Tensor): Reference points for sampling.
                    - Shape: [bs, num_query, num_levels, 2] for 2D coordinates
                    - Shape: [bs, num_query, num_levels, 4] for boxes (x, y, w, h)
                    - Values normalized to [0, 1] range
                spatial_shapes (torch.Tensor): Spatial dimensions for each feature level.
                    - Shape: [num_levels, 2]
                    - Each row: [H_l, W_l] for level l
                **kwargs: Additional keyword arguments (unused, kept for interface compatibility).
                    - May include: key, level_start_index, flag, etc.
                    - These are ignored but accepted to maintain compatibility with callers.
                    
            Returns:
                torch.Tensor: Attention output with residual connection.
                    - Shape: [num_query, bs, embed_dims] (sequence-first format)
                    - Same format as input query
            """
            value = value if value is not None else query
            identity = identity if identity is not None else query
            
            if query_pos is not None:
                query = query + query_pos
            
            # Convert to batch-first: [num_query, bs, embed_dims] -> [bs, num_query, embed_dims]
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)
            
            bs, num_query = query.shape[:2]
            num_value = value.shape[1]
            
            total_spatial_size = (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum()
            assert total_spatial_size == num_value, \
                f'spatial_shapes sum {total_spatial_size} != num_value {num_value}'
            
            # Project value: [bs, num_value, embed_dims] -> [bs, num_value, embed_dims]
            value = self.value_proj(value)
            if key_padding_mask is not None:
                value = value.masked_fill(key_padding_mask[..., None], 0.0)
            
            # Reshape for multi-head: [bs, num_value, embed_dims] -> [bs, num_value, num_heads, dim_per_head]
            value = value.view(bs, num_value, self.num_heads, self.dim_per_head)
            
            # Compute offsets: [bs, num_query, embed_dims] -> [bs, num_query, num_heads, num_levels, num_points, 2]
            sampling_offsets = self.sampling_offsets(query).view(
                bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
            
            # Compute weights: [bs, num_query, embed_dims] -> [bs, num_query, num_heads, num_levels*num_points]
            attention_weights = self.attention_weights(query).view(
                bs, num_query, self.num_heads, self.num_levels * self.num_points)
            attention_weights = attention_weights.softmax(dim=-1)
            attention_weights = attention_weights.view(
                bs, num_query, self.num_heads, self.num_levels, self.num_points)
            
            # Compute locations: [bs, num_query, num_heads, num_levels, num_points, 2]
            sampling_locations = self._compute_sampling_locations(
                reference_points, sampling_offsets, spatial_shapes)
            
            # Apply attention
            if _has_mmcv_deform_attn:
                output = multi_scale_deformable_attn_pytorch(
                    value, spatial_shapes, sampling_locations, attention_weights)
            else:
                output = self._deformable_attn_pytorch(
                    value, spatial_shapes, sampling_locations, attention_weights)
            
            # Reshape and project: [bs, num_query, num_heads, dim_per_head] -> [bs, num_query, embed_dims]
            output = output.reshape(bs, num_query, self.embed_dims)
            output = self.output_proj(output)
            
            # Convert back: [bs, num_query, embed_dims] -> [num_query, bs, embed_dims]
            output = output.permute(1, 0, 2)
            
            return self.dropout(output) + identity

