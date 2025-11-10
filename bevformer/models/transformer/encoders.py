"""BEVFormer Encoder and Layer implementation, aligned with original BEVFormer."""
import copy
import warnings
import numpy as np
from typing import Optional, List
import torch
import torch.nn as nn

try:
    from mmdet.models.utils.transformer import TransformerLayerSequence
    from mmcv.runner import force_fp32, auto_fp16
    from mmcv.utils import TORCH_VERSION, digit_version
except Exception:
    TransformerLayerSequence = nn.Module
    def force_fp32(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    def auto_fp16(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    TORCH_VERSION = "1.0.0"
    def digit_version(version):
        return [0, 0, 0]

try:
    from mmdet.models.utils.builder import TRANSFORMER_LAYER_SEQUENCE, TRANSFORMER_LAYER
except Exception:
    try:
        from mmdet.models.builder import TRANSFORMER_LAYER_SEQUENCE, TRANSFORMER_LAYER
    except Exception:
        TRANSFORMER_LAYER_SEQUENCE = None
        TRANSFORMER_LAYER = None

from .custom_base_transformer_layer import MyCustomBaseTransformerLayer

if TRANSFORMER_LAYER_SEQUENCE is not None:
    @TRANSFORMER_LAYER_SEQUENCE.register_module()
    class BEVFormerEncoder(TransformerLayerSequence):
        """BEVFormer encoder that stacks multiple BEVFormerLayer blocks.
        
        This encoder implements the same logic as the original BEVFormer:
        - Generates 3D and 2D reference points
        - Projects 3D points to camera views
        - Builds hybrid reference points for temporal attention
        - Passes all necessary parameters to layers
        """
        
        def __init__(self, 
                     num_layers: int = 6,
                     embed_dims: int = 256,
                     pc_range: Optional[List[float]] = None,
                     num_points_in_pillar: int = 4,
                     return_intermediate: bool = False,
                     transformerlayers: Optional[dict] = None,
                     **kwargs):
            super().__init__()
            self.num_layers = num_layers
            self.embed_dims = embed_dims
            self.pc_range = pc_range or [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
            self.num_points_in_pillar = num_points_in_pillar
            self.return_intermediate = return_intermediate
            self.fp16_enabled = False
            
            # Build layers from config
            if transformerlayers is not None:
                from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
                # Use mmcv's builder to construct layers
                layer_cfg = transformerlayers.copy()
                layer_cfg['num_layers'] = num_layers
                self.layers = build_transformer_layer_sequence(layer_cfg)
            else:
                # Fallback: create default layers
                self.layers = nn.ModuleList([
                    BEVFormerLayer(
                        attn_cfgs=[
                            dict(type='TemporalSelfAttention', embed_dims=embed_dims, num_levels=2, num_points=8, num_heads=8),
                            dict(type='SpatialCrossAttention', embed_dims=embed_dims, num_cams=6, pc_range=self.pc_range)
                        ],
                        feedforward_channels=embed_dims * 2,
                        ffn_dropout=0.1,
                        operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')
                    ) for _ in range(num_layers)
                ])
        
        @staticmethod
        def get_reference_points(H, W, Z=8, num_points_in_pillar=4, dim='3d', bs=1, device='cuda', dtype=torch.float):
            """Get the reference points used in SCA and TSA.
            
            Args:
                H, W: spatial shape of bev
                Z: height of pillar
                num_points_in_pillar: sample D points uniformly from each pillar
                dim: '3d' for spatial cross-attention, '2d' for temporal self-attention
                device: device to create tensors on
                dtype: data type
                
            Returns:
                Tensor: reference points
            """
            # Reference points in 3D space, used in spatial cross-attention (SCA)
            if dim == '3d':
                zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar, dtype=dtype,
                                    device=device).view(-1, 1, 1).expand(num_points_in_pillar, H, W) / Z
                xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
                                    device=device).view(1, 1, W).expand(num_points_in_pillar, H, W) / W
                ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
                                    device=device).view(1, H, 1).expand(num_points_in_pillar, H, W) / H
                ref_3d = torch.stack((xs, ys, zs), -1)
                ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
                ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)
                return ref_3d
            
            # Reference points on 2D BEV plane, used in temporal self-attention (TSA)
            elif dim == '2d':
                ref_y, ref_x = torch.meshgrid(
                    torch.linspace(0.5, H - 0.5, H, dtype=dtype, device=device),
                    torch.linspace(0.5, W - 0.5, W, dtype=dtype, device=device),
                    indexing='ij'
                )
                ref_y = ref_y.reshape(-1)[None] / H
                ref_x = ref_x.reshape(-1)[None] / W
                ref_2d = torch.stack((ref_x, ref_y), -1)
                ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
                return ref_2d
        
        @force_fp32(apply_to=('reference_points', 'img_metas'))
        def point_sampling(self, reference_points, pc_range, img_metas):
            """Project 3D reference points to camera views.
            
            This function must use fp32 for numerical stability.
            
            Args:
                reference_points: [B, N, D, 3] 3D reference points
                pc_range: point cloud range
                img_metas: list of image meta information
                
            Returns:
                reference_points_cam: [num_cam, B, N, D, 2] projected points
                bev_mask: [num_cam, B, N, D] visibility mask
            """
            # Close tf32 for numerical stability
            allow_tf32 = torch.backends.cuda.matmul.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            
            lidar2img = []
            for img_meta in img_metas:
                lidar2img.append(img_meta['lidar2img'])
            lidar2img = np.asarray(lidar2img)
            lidar2img = reference_points.new_tensor(lidar2img)  # (B, N, 4, 4)
            reference_points = reference_points.clone()
            
            # Convert normalized coordinates to real world coordinates
            reference_points[..., 0:1] = reference_points[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
            reference_points[..., 1:2] = reference_points[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
            reference_points[..., 2:3] = reference_points[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2]
            
            # Add homogeneous coordinate
            reference_points = torch.cat(
                (reference_points, torch.ones_like(reference_points[..., :1])), -1)
            
            reference_points = reference_points.permute(1, 0, 2, 3)
            D, B, num_query = reference_points.size()[:3]
            num_cam = lidar2img.size(1)
            
            reference_points = reference_points.view(
                D, B, 1, num_query, 4).repeat(1, 1, num_cam, 1, 1).unsqueeze(-1)
            
            lidar2img = lidar2img.view(
                1, B, num_cam, 1, 4, 4).repeat(D, 1, 1, num_query, 1, 1)
            
            # Project to camera views
            reference_points_cam = torch.matmul(lidar2img.to(torch.float32),
                                                reference_points.to(torch.float32)).squeeze(-1)
            eps = 1e-5
            
            bev_mask = (reference_points_cam[..., 2:3] > eps)
            reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
                reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3]) * eps)
            
            # Normalize to [0, 1]
            reference_points_cam[..., 0] /= img_metas[0]['img_shape'][0][1]
            reference_points_cam[..., 1] /= img_metas[0]['img_shape'][0][0]
            
            # Visibility mask
            bev_mask = (bev_mask & (reference_points_cam[..., 1:2] > 0.0)
                        & (reference_points_cam[..., 1:2] < 1.0)
                        & (reference_points_cam[..., 0:1] < 1.0)
                        & (reference_points_cam[..., 0:1] > 0.0))
            if digit_version(TORCH_VERSION) >= digit_version('1.8'):
                bev_mask = torch.nan_to_num(bev_mask)
            else:
                bev_mask = bev_mask.new_tensor(
                    np.nan_to_num(bev_mask.cpu().numpy()))
            
            reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4)
            bev_mask = bev_mask.permute(2, 1, 3, 0, 4).squeeze(-1)
            
            # Restore tf32
            torch.backends.cuda.matmul.allow_tf32 = allow_tf32
            torch.backends.cudnn.allow_tf32 = allow_tf32
            
            return reference_points_cam, bev_mask
        
        @auto_fp16()
        def forward(self,
                    bev_query,
                    key,
                    value,
                    *args,
                    bev_h=None,
                    bev_w=None,
                    bev_pos=None,
                    spatial_shapes=None,
                    level_start_index=None,
                    valid_ratios=None,
                    prev_bev=None,
                    shift=0.,
                    **kwargs):
            """Forward function for BEVFormerEncoder.
            
            Args:
                bev_query: [num_query, bs, embed_dims] or [bs, num_query, embed_dims]
                key, value: [num_cam, num_value, bs, embed_dims] multi-camera features
                bev_h, bev_w: BEV spatial dimensions
                bev_pos: [num_query, bs, embed_dims] or [bs, num_query, embed_dims] positional encoding
                prev_bev: [num_query, bs, embed_dims] or [bs, num_query, embed_dims] previous BEV
                shift: [bs, 2] shift for temporal alignment
                
            Returns:
                output: [bs, num_query, embed_dims] encoded BEV features
            """
            output = bev_query
            intermediate = []
            
            # Generate reference points
            ref_3d = self.get_reference_points(
                bev_h, bev_w, self.pc_range[5]-self.pc_range[2], 
                self.num_points_in_pillar, dim='3d', 
                bs=bev_query.size(1) if bev_query.dim() == 3 else bev_query.size(0),
                device=bev_query.device, dtype=bev_query.dtype)
            ref_2d = self.get_reference_points(
                bev_h, bev_w, dim='2d', 
                bs=bev_query.size(1) if bev_query.dim() == 3 else bev_query.size(0),
                device=bev_query.device, dtype=bev_query.dtype)
            
            # Project 3D points to camera views
            reference_points_cam, bev_mask = self.point_sampling(
                ref_3d, self.pc_range, kwargs['img_metas'])
            
            # Apply shift to ref_2d for temporal alignment
            shift_ref_2d = ref_2d.clone()
            if shift is not None and isinstance(shift, torch.Tensor):
                shift_ref_2d += shift[:, None, None, :]
            
            # Convert to batch_first format if needed
            if bev_query.dim() == 3 and bev_query.size(0) != bev_query.size(1):
                # [num_query, bs, embed_dims] -> [bs, num_query, embed_dims]
                bev_query = bev_query.permute(1, 0, 2)
                if bev_pos is not None:
                    bev_pos = bev_pos.permute(1, 0, 2)
                if prev_bev is not None:
                    prev_bev = prev_bev.permute(1, 0, 2)
            
            bs, len_bev, num_bev_level, _ = ref_2d.shape
            
            # Build hybrid reference points for temporal attention
            if prev_bev is not None:
                prev_bev = torch.stack([prev_bev, bev_query], 1).reshape(bs*2, len_bev, -1)
                hybird_ref_2d = torch.stack([shift_ref_2d, ref_2d], 1).reshape(
                    bs*2, len_bev, num_bev_level, 2)
            else:
                hybird_ref_2d = torch.stack([ref_2d, ref_2d], 1).reshape(
                    bs*2, len_bev, num_bev_level, 2)
                prev_bev = None
            
            # Process through layers
            for lid, layer in enumerate(self.layers):
                output = layer(
                    bev_query,
                    key,
                    value,
                    *args,
                    bev_pos=bev_pos,
                    ref_2d=hybird_ref_2d,
                    ref_3d=ref_3d,
                    bev_h=bev_h,
                    bev_w=bev_w,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    reference_points_cam=reference_points_cam,
                    bev_mask=bev_mask,
                    prev_bev=prev_bev,
                    **kwargs)
                
                bev_query = output
                if self.return_intermediate:
                    intermediate.append(output)
            
            if self.return_intermediate:
                return torch.stack(intermediate)
            
            return output


if TRANSFORMER_LAYER is not None:
    @TRANSFORMER_LAYER.register_module()
    class BEVFormerLayer(MyCustomBaseTransformerLayer):
        """BEVFormer layer that integrates TemporalSelfAttention and SpatialCrossAttention.
        
        This layer follows the original BEVFormer design:
        - self_attn: TemporalSelfAttention (fuses prev_bev and current_bev)
        - cross_attn: SpatialCrossAttention (aggregates multi-view image features)
        - ffn: Feed-forward network
        """
        
        def __init__(self,
                     attn_cfgs,
                     feedforward_channels,
                     ffn_dropout=0.0,
                     operation_order=None,
                     act_cfg=dict(type='ReLU', inplace=True),
                     norm_cfg=dict(type='LN'),
                     ffn_num_fcs=2,
                     **kwargs):
            super(BEVFormerLayer, self).__init__(
                attn_cfgs=attn_cfgs,
                feedforward_channels=feedforward_channels,
                ffn_dropout=ffn_dropout,
                operation_order=operation_order,
                act_cfg=act_cfg,
                norm_cfg=norm_cfg,
                ffn_num_fcs=ffn_num_fcs,
                **kwargs)
            self.fp16_enabled = False
            assert len(operation_order) == 6
            assert set(operation_order) == set(['self_attn', 'norm', 'cross_attn', 'ffn'])
        
        def forward(self,
                    query,
                    key=None,
                    value=None,
                    bev_pos=None,
                    query_pos=None,
                    key_pos=None,
                    attn_masks=None,
                    query_key_padding_mask=None,
                    key_padding_mask=None,
                    ref_2d=None,
                    ref_3d=None,
                    bev_h=None,
                    bev_w=None,
                    reference_points_cam=None,
                    mask=None,
                    spatial_shapes=None,
                    level_start_index=None,
                    prev_bev=None,
                    **kwargs):
            """Forward function for BEVFormerLayer.
            
            Args:
                query: [bs, num_query, embed_dims] BEV queries
                key, value: [num_cam, num_value, bs, embed_dims] multi-camera features
                bev_pos: [bs, num_query, embed_dims] positional encoding
                ref_2d: [bs*2, num_query, num_levels, 2] 2D reference points for TSA
                ref_3d: [bs, num_query, D, 3] 3D reference points for SCA
                reference_points_cam: [num_cam, bs, num_query, D, 2] projected points
                bev_mask: [num_cam, bs, num_query, D] visibility mask
                prev_bev: [bs*2, num_query, embed_dims] previous BEV (stacked with current)
                
            Returns:
                query: [bs, num_query, embed_dims] updated BEV queries
            """
            norm_index = 0
            attn_index = 0
            ffn_index = 0
            identity = query
            
            if attn_masks is None:
                attn_masks = [None for _ in range(self.num_attn)]
            elif isinstance(attn_masks, torch.Tensor):
                attn_masks = [copy.deepcopy(attn_masks) for _ in range(self.num_attn)]
            else:
                assert len(attn_masks) == self.num_attn
            
            for layer in self.operation_order:
                # Temporal self-attention
                if layer == 'self_attn':
                    query = self.attentions[attn_index](
                        query,
                        prev_bev,  # key
                        prev_bev,  # value
                        identity if self.pre_norm else None,  # residual
                        query_pos=bev_pos,
                        key_pos=bev_pos,
                        attn_mask=attn_masks[attn_index],
                        key_padding_mask=query_key_padding_mask,
                        reference_points=ref_2d,
                        spatial_shapes=torch.tensor(
                            [[bev_h, bev_w]], device=query.device, dtype=torch.long),
                        level_start_index=torch.tensor([0], device=query.device, dtype=torch.long),
                        **kwargs)
                    attn_index += 1
                    identity = query
                
                elif layer == 'norm':
                    query = self.norms[norm_index](query)
                    norm_index += 1
                
                # Spatial cross-attention
                elif layer == 'cross_attn':
                    query = self.attentions[attn_index](
                        query,
                        key,
                        value,
                        identity if self.pre_norm else None,  # residual
                        query_pos=query_pos if query_pos is not None else bev_pos,
                        key_pos=key_pos,
                        reference_points=ref_3d,
                        reference_points_cam=reference_points_cam,
                        mask=mask,
                        attn_mask=attn_masks[attn_index],
                        key_padding_mask=key_padding_mask,
                        spatial_shapes=spatial_shapes,
                        level_start_index=level_start_index,
                        bev_mask=mask,  # bev_mask is passed as mask
                        **kwargs)
                    attn_index += 1
                    identity = query
                
                elif layer == 'ffn':
                    query = self.ffns[ffn_index](
                        query, identity if self.pre_norm else None)
                    ffn_index += 1
            
            return query
