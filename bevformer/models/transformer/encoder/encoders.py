"""BEVFormer Encoder implementation."""
import numpy as np
from typing import Optional, List
import torch
from mmdet.models.utils.transformer import TransformerLayerSequence
from mmcv.runner import force_fp32, auto_fp16
from mmcv.utils import TORCH_VERSION, digit_version
from mmdet.models.utils.builder import TRANSFORMER_LAYER_SEQUENCE

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
                     *args,
                     pc_range: Optional[List[float]] = None,
                     num_points_in_pillar: int = 4,
                     return_intermediate: bool = False,
                     **kwargs):
            super(BEVFormerEncoder, self).__init__(*args, **kwargs)
            self.return_intermediate = return_intermediate
            self.num_points_in_pillar = num_points_in_pillar
            self.pc_range = pc_range or [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
            self.fp16_enabled = False
        
        @staticmethod
        def get_reference_points(H, W, Z=8, num_points_in_pillar=4, dim='3d', bs=1, device='cuda', dtype=torch.float):
            """Get reference points for SCA (3D) or TSA (2D).
            
            Args:
                H, W: BEV spatial dimensions
                Z: pillar height
                num_points_in_pillar: points per pillar
                dim: '3d' for SCA, '2d' for TSA
                bs: batch size
                device, dtype: tensor device and dtype
                
            Returns:
                [bs, N, D, 3] for 3D or [bs, N, 1, 2] for 2D
            """
            if dim == '3d':
                # Generate 3D grid: [num_points_in_pillar, H, W, 3]
                zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar, dtype=dtype, device=device)
                zs = zs.view(-1, 1, 1).expand(num_points_in_pillar, H, W) / Z
                xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype, device=device)
                xs = xs.view(1, 1, W).expand(num_points_in_pillar, H, W) / W
                ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype, device=device)
                ys = ys.view(1, H, 1).expand(num_points_in_pillar, H, W) / H
                ref_3d = torch.stack((xs, ys, zs), -1)
                # Reshape: [num_points_in_pillar, H*W, 3] -> [bs, H*W, num_points_in_pillar, 3]
                ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
                return ref_3d[None].repeat(bs, 1, 1, 1)
            else:  # dim == '2d'
                # Generate 2D grid: [H, W] -> [H*W, 2]
                ref_y, ref_x = torch.meshgrid(
                    torch.linspace(0.5, H - 0.5, H, dtype=dtype, device=device),
                    torch.linspace(0.5, W - 0.5, W, dtype=dtype, device=device),
                    indexing='ij')
                ref_2d = torch.stack((ref_x.reshape(-1) / W, ref_y.reshape(-1) / H), -1)
                return ref_2d[None].repeat(bs, 1, 1).unsqueeze(2)
        
        @force_fp32(apply_to=('reference_points', 'img_metas'))
        def point_sampling(self, reference_points, pc_range, img_metas):
            """Project 3D reference points to camera views.
            
            This function must use fp32 for numerical stability in coordinate transformations.
            
            Args:
                reference_points: 3D reference points in normalized coordinates
                    - Shape: [B, N, D, 3]
                    - Format: (x, y, z) where each coordinate is in [0, 1]
                    - N: number of BEV queries (bev_h * bev_w)
                    - D: number of points per pillar (num_points_in_pillar, typically 4)
                pc_range: Point cloud range [x_min, y_min, z_min, x_max, y_max, z_max]
                    - Used to convert normalized coordinates to real-world coordinates
                img_metas: List[dict] of image meta information
                    - Each dict contains 'lidar2img': [num_cam, 4, 4] transformation matrix
                    - Each dict contains 'img_shape': [[H, W]] image dimensions
                
            Returns:
                reference_points_cam: Projected 2D points in camera views
                    - Shape: [num_cam, B, N, D, 2]
                    - Format: (u, v) normalized to [0, 1] range
                bev_mask: Visibility mask indicating which points are visible in each camera
                    - Shape: [num_cam, B, N, D]
                    - True if point is visible (in front of camera and within image bounds)
            """
            # Close tf32 for numerical stability in matrix multiplications
            allow_tf32 = torch.backends.cuda.matmul.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            
            # Extract lidar2img: [B, num_cam, 4, 4]
            lidar2img = reference_points.new_tensor(
                np.asarray([img_meta['lidar2img'] for img_meta in img_metas]))
            reference_points = reference_points.clone()
            
            # Convert [0, 1] normalized coords to real-world coords
            reference_points[..., 0] = reference_points[..., 0] * (pc_range[3] - pc_range[0]) + pc_range[0]
            reference_points[..., 1] = reference_points[..., 1] * (pc_range[4] - pc_range[1]) + pc_range[1]
            reference_points[..., 2] = reference_points[..., 2] * (pc_range[5] - pc_range[2]) + pc_range[2]
            
            # Add homogeneous coordinate (w=1) for matrix multiplication
            # reference_points: [B, N, D, 3] -> [B, N, D, 4]
            reference_points = torch.cat(
                (reference_points, torch.ones_like(reference_points[..., :1])), -1)
            
            # Reshape for broadcasting: [B, N, D, 4] -> [D, B, N, 4]
            reference_points = reference_points.permute(2, 0, 1, 3)
            D, B, num_query, _ = reference_points.shape
            num_cam = lidar2img.size(1)
            
            # Expand for all cameras: [D, B, num_cam, N, 4]
            reference_points = reference_points.view(D, B, 1, num_query, 4).repeat(1, 1, num_cam, 1, 1)
            lidar2img = lidar2img.view(1, B, num_cam, 1, 4, 4).repeat(D, 1, 1, num_query, 1, 1)
            
            # Project: lidar2img @ [D, B, num_cam, N, 4, 1] -> [D, B, num_cam, N, 4]
            reference_points_cam = torch.matmul(
                lidar2img.to(torch.float32), reference_points.unsqueeze(-1).to(torch.float32)).squeeze(-1)
            
            # Check if points are in front of camera (z > 0)
            eps = 1e-5
            bev_mask = (reference_points_cam[..., 2:3] > eps)
            
            # Perspective division: [D, B, num_cam, N, 2]
            z = torch.maximum(reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3]) * eps)
            reference_points_cam = reference_points_cam[..., 0:2] / z
            
            # Normalize to [0, 1]: divide by image dimensions
            img_shape = img_metas[0]['img_shape'][0]  # [H, W]
            reference_points_cam[..., 0] /= img_shape[1]  # width
            reference_points_cam[..., 1] /= img_shape[0]  # height
            
            # Visibility mask: in front of camera AND within image bounds
            bev_mask = (bev_mask &
                        (reference_points_cam[..., 0:1] > 0.0) & (reference_points_cam[..., 0:1] < 1.0) &
                        (reference_points_cam[..., 1:2] > 0.0) & (reference_points_cam[..., 1:2] < 1.0))
            
            # Handle NaN values
            if digit_version(TORCH_VERSION) >= digit_version('1.8'):
                bev_mask = torch.nan_to_num(bev_mask)
            else:
                bev_mask = bev_mask.new_tensor(np.nan_to_num(bev_mask.cpu().numpy()))
            
            # Permute to [num_cam, B, N, D, 2] and [num_cam, B, N, D]
            reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4)
            bev_mask = bev_mask.permute(2, 1, 3, 0, 4).squeeze(-1)
            
            # Restore tf32 settings
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
            # Get batch size and convert to batch-first format
            bs = bev_query.size(1) if bev_query.dim() == 3 else bev_query.size(0)
            bev_query = bev_query.permute(1, 0, 2) if bev_query.dim() == 3 else bev_query
            bev_pos = bev_pos.permute(1, 0, 2) if bev_pos.dim() == 3 else bev_pos
            
            # Generate reference points
            ref_3d = self.get_reference_points(
                bev_h, bev_w, self.pc_range[5] - self.pc_range[2],
                self.num_points_in_pillar, dim='3d', bs=bs,
                device=bev_query.device, dtype=bev_query.dtype)
            ref_2d = self.get_reference_points(
                bev_h, bev_w, dim='2d', bs=bs,
                device=bev_query.device, dtype=bev_query.dtype)
            
            # Project 3D points to camera views
            reference_points_cam, bev_mask = self.point_sampling(
                ref_3d, self.pc_range, kwargs['img_metas'])
            
            # Apply shift for temporal alignment
            shift_ref_2d = ref_2d.clone()
            shift_ref_2d += shift[:, None, None, :]
            
            # Build hybrid reference points: [bs*2, N, 1, 2]
            len_bev, num_bev_level = ref_2d.shape[1], ref_2d.shape[2]
            if prev_bev is not None:
                prev_bev = prev_bev.permute(1, 0, 2) if prev_bev.dim() == 3 else prev_bev
                prev_bev = torch.stack([prev_bev, bev_query], 1).reshape(bs * 2, len_bev, -1)
                hybird_ref_2d = torch.stack([shift_ref_2d, ref_2d], 1).reshape(bs * 2, len_bev, num_bev_level, 2)
            else:
                hybird_ref_2d = torch.stack([ref_2d, ref_2d], 1).reshape(bs * 2, len_bev, num_bev_level, 2)
                prev_bev = None
            
            # Process through layers
            intermediate = []
            for layer in self.layers:
                bev_query = layer(
                    bev_query, key, value, *args,
                    bev_pos=bev_pos, ref_2d=hybird_ref_2d, ref_3d=ref_3d,
                    bev_h=bev_h, bev_w=bev_w,
                    spatial_shapes=spatial_shapes, level_start_index=level_start_index,
                    reference_points_cam=reference_points_cam, bev_mask=bev_mask,
                    prev_bev=prev_bev, **kwargs)
                if self.return_intermediate:
                    intermediate.append(bev_query)
            
            return torch.stack(intermediate) if self.return_intermediate else bev_query


