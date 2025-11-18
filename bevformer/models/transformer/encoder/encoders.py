"""BEVFormer Encoder implementation.

This module implements the BEVFormer encoder, which is responsible for:
1. Generating 3D reference points for spatial cross-attention (SCA)
2. Generating 2D reference points for temporal self-attention (TSA)
3. Projecting 3D reference points to camera views
4. Building hybrid reference points for temporal alignment
5. Coordinating the forward pass through multiple BEVFormerLayer blocks
"""
import numpy as np
from typing import Optional, List
import torch
from mmdet.models.utils.transformer import TransformerLayerSequence
from mmcv.runner import force_fp32, auto_fp16
from mmcv.utils import TORCH_VERSION, digit_version
from mmcv.cnn.bricks.registry import TRANSFORMER_LAYER_SEQUENCE

if TRANSFORMER_LAYER_SEQUENCE is not None:
    @TRANSFORMER_LAYER_SEQUENCE.register_module()
    class BEVFormerEncoder(TransformerLayerSequence):
        """BEVFormer encoder that stacks multiple BEVFormerLayer blocks.
        
        This encoder implements the same logic as the original BEVFormer:
        - Generates 3D and 2D reference points
        - Projects 3D points to camera views
        - Builds hybrid reference points for temporal attention
        - Passes all necessary parameters to layers
        
        The encoder processes BEV queries through multiple transformer layers,
        where each layer performs:
        1. Temporal Self-Attention (TSA): Aggregates information from previous BEV features
        2. Spatial Cross-Attention (SCA): Aggregates information from multi-camera features
        
        Args:
            pc_range (List[float]): Point cloud range [x_min, y_min, z_min, x_max, y_max, z_max]
                Used for coordinate transformations between normalized and real-world coordinates.
            num_points_in_pillar (int): Number of height anchors (Z anchors) per BEV query.
                Each BEV query has multiple 3D reference points at different heights.
                Default: 4
            return_intermediate (bool): Whether to return intermediate outputs from all layers.
                If True, returns a tensor of shape [num_layers, bs, num_query, embed_dims].
                If False, returns only the final output of shape [bs, num_query, embed_dims].
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
            
            This function generates reference points that are used to sample features:
            - For 3D (SCA): Generates points at different heights (Z anchors) for each BEV query
            - For 2D (TSA): Generates 2D grid points for temporal attention
            
            Args:
                H (int): BEV height (number of queries along Y-axis)
                W (int): BEV width (number of queries along X-axis)
                Z (int): Pillar height in meters. Used to determine the range of Z coordinates.
                    Default: 8
                num_points_in_pillar (int): Number of height anchors per BEV query.
                    Each BEV query has multiple 3D reference points at different heights.
                    Default: 4
                dim (str): '3d' for spatial cross-attention (SCA) or '2d' for temporal self-attention (TSA)
                bs (int): Batch size. Default: 1
                device (str): Device to create tensors on. Default: 'cuda'
                dtype (torch.dtype): Data type for tensors. Default: torch.float
                
            Returns:
                torch.Tensor: Reference points
                    - For 3D: Shape [bs, H*W, num_points_in_pillar, 3]
                        - Each point is (x, y, z) in normalized coordinates [0, 1]
                        - H*W is the total number of BEV queries
                        - num_points_in_pillar is the number of height anchors
                    - For 2D: Shape [bs, H*W, 1, 2]
                        - Each point is (x, y) in normalized coordinates [0, 1]
                        - The third dimension is 1 for compatibility with 3D format
            """
            if dim == '3d':
                # Generate 3D reference points for spatial cross-attention (SCA)
                # Each BEV query has num_points_in_pillar reference points at different heights
                
                # Generate Z coordinates: [num_points_in_pillar] evenly spaced from 0.5 to Z-0.5
                # Then normalize to [0, 1] range
                zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar, dtype=dtype, device=device)
                zs = zs.view(-1, 1, 1).expand(num_points_in_pillar, H, W) / Z
                
                # Generate X coordinates: [W] evenly spaced from 0.5 to W-0.5, normalized to [0, 1]
                xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype, device=device)
                xs = xs.view(1, 1, W).expand(num_points_in_pillar, H, W) / W
                
                # Generate Y coordinates: [H] evenly spaced from 0.5 to H-0.5, normalized to [0, 1]
                ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype, device=device)
                ys = ys.view(1, H, 1).expand(num_points_in_pillar, H, W) / H
                
                # Stack coordinates: [num_points_in_pillar, H, W, 3]
                ref_3d = torch.stack((xs, ys, zs), -1)
                
                # Reshape to [bs, H*W, num_points_in_pillar, 3]
                # permute(0, 3, 1, 2): [num_points_in_pillar, H, W, 3] -> [num_points_in_pillar, 3, H, W]
                # flatten(2): [num_points_in_pillar, 3, H, W] -> [num_points_in_pillar, 3, H*W]
                # permute(2, 0, 1): [num_points_in_pillar, 3, H*W] -> [H*W, num_points_in_pillar, 3]
                ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(2, 0, 1)
                # Add batch dimension: [H*W, num_points_in_pillar, 3] -> [bs, H*W, num_points_in_pillar, 3]
                return ref_3d[None].repeat(bs, 1, 1, 1)
            else:  # dim == '2d'
                # Generate 2D reference points for temporal self-attention (TSA)
                # These are used to align current and previous BEV features
                
                # Create meshgrid: [H, W] -> [H, W] for both x and y
                ref_y, ref_x = torch.meshgrid(
                    torch.linspace(0.5, H - 0.5, H, dtype=dtype, device=device),
                    torch.linspace(0.5, W - 0.5, W, dtype=dtype, device=device),
                    indexing='ij')
                
                # Normalize to [0, 1] and reshape: [H*W, 2]
                ref_2d = torch.stack((ref_x.reshape(-1) / W, ref_y.reshape(-1) / H), -1)
                
                # Add batch and level dimensions: [bs, H*W, 1, 2]
                # The third dimension (1) is for compatibility with multi-level features
                return ref_2d[None].repeat(bs, 1, 1).unsqueeze(2)
        
        @force_fp32(apply_to=('reference_points', 'img_metas'))
        def point_sampling(self, reference_points, pc_range, img_metas):
            """Project 3D reference points to camera views.
            
            This function projects 3D reference points from LiDAR coordinate system to 2D image
            coordinates in each camera view. It also computes visibility masks indicating which
            points are visible in each camera.
            
            IMPORTANT: This function must use fp32 for numerical stability in coordinate
            transformations. Matrix multiplications and coordinate conversions are sensitive
            to precision, so fp16 is disabled here.
            
            The projection process:
            1. Convert normalized coordinates [0, 1] to real-world coordinates using pc_range
            2. Add homogeneous coordinate (w=1) for matrix multiplication
            3. Transform from LiDAR to camera coordinates using lidar2img matrices
            4. Perform perspective division to get 2D image coordinates
            5. Normalize to [0, 1] range using image dimensions
            6. Compute visibility mask (in front of camera AND within image bounds)
            
            Args:
                reference_points (torch.Tensor): 3D reference points in normalized coordinates
                    - Shape: [B, N, D, 3]
                    - Format: (x, y, z) where each coordinate is in [0, 1]
                    - B: batch size
                    - N: number of BEV queries (bev_h * bev_w)
                    - D: number of height anchors per query (num_points_in_pillar, typically 4)
                pc_range (List[float]): Point cloud range [x_min, y_min, z_min, x_max, y_max, z_max]
                    - Used to convert normalized coordinates [0, 1] to real-world coordinates
                    - Example: [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0] for NuScenes
                img_metas (List[dict]): List of image meta information, one per batch item
                    - Each dict contains:
                        - 'lidar2img': [num_cam, 4, 4] transformation matrix from LiDAR to image
                        - 'img_shape': [[H, W]] image dimensions (height, width)
                    - num_cam: number of cameras (typically 6 for NuScenes)
                
            Returns:
                tuple: (reference_points_cam, bev_mask)
                    reference_points_cam (torch.Tensor): Projected 2D points in camera views
                        - Shape: [num_cam, B, N, D, 2]
                        - Format: (u, v) normalized to [0, 1] range
                        - u: horizontal coordinate (0 = left, 1 = right)
                        - v: vertical coordinate (0 = top, 1 = bottom)
                    bev_mask (torch.Tensor): Visibility mask indicating which points are visible
                        - Shape: [num_cam, B, N, D]
                        - dtype: bool or float (True/1.0 if visible, False/0.0 if not)
                        - A point is visible if:
                            - It is in front of the camera (z > 0 after projection)
                            - It is within image bounds (0 < u < 1 and 0 < v < 1)
            """
            # Disable tf32 for numerical stability in matrix multiplications
            # tf32 can cause precision issues in coordinate transformations
            allow_tf32 = torch.backends.cuda.matmul.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            
            # Extract lidar2img transformation matrices: [B, num_cam, 4, 4]
            # Each matrix transforms points from LiDAR coordinate system to camera coordinate system
            lidar2img = reference_points.new_tensor(
                np.asarray([img_meta['lidar2img'] for img_meta in img_metas]))
            reference_points = reference_points.clone()  # Avoid modifying input
            
            # Step 1: Convert normalized coordinates [0, 1] to real-world coordinates
            # x: [0, 1] -> [pc_range[0], pc_range[3]]
            reference_points[..., 0] = reference_points[..., 0] * (pc_range[3] - pc_range[0]) + pc_range[0]
            # y: [0, 1] -> [pc_range[1], pc_range[4]]
            reference_points[..., 1] = reference_points[..., 1] * (pc_range[4] - pc_range[1]) + pc_range[1]
            # z: [0, 1] -> [pc_range[2], pc_range[5]]
            reference_points[..., 2] = reference_points[..., 2] * (pc_range[5] - pc_range[2]) + pc_range[2]
            
            # Step 2: Add homogeneous coordinate (w=1) for matrix multiplication
            # reference_points: [B, N, D, 3] -> [B, N, D, 4]
            # Homogeneous coordinates allow us to represent translations as matrix multiplications
            reference_points = torch.cat(
                (reference_points, torch.ones_like(reference_points[..., :1])), -1)
            
            # Step 3: Reshape for efficient broadcasting across cameras
            # [B, N, D, 4] -> [D, B, N, 4]
            # We move D (height anchors) to the first dimension for easier broadcasting
            reference_points = reference_points.permute(2, 0, 1, 3)
            D, B, num_query, _ = reference_points.shape
            num_cam = lidar2img.size(1)  # Number of cameras
            
            # Step 4: Expand for all cameras
            # reference_points: [D, B, 1, N, 4] -> [D, B, num_cam, N, 4]
            # Each height anchor is now replicated for each camera
            reference_points = reference_points.view(D, B, 1, num_query, 4).repeat(1, 1, num_cam, 1, 1)
            # lidar2img: [1, B, num_cam, 1, 4, 4] -> [D, B, num_cam, N, 4, 4]
            # Each transformation matrix is replicated for each height anchor and query
            lidar2img = lidar2img.view(1, B, num_cam, 1, 4, 4).repeat(D, 1, 1, num_query, 1, 1)
            
            # Step 5: Project points from LiDAR to camera coordinates
            # lidar2img @ reference_points: [D, B, num_cam, N, 4, 4] @ [D, B, num_cam, N, 4, 1] -> [D, B, num_cam, N, 4]
            # Result is in camera coordinate system: (x_cam, y_cam, z_cam, w)
            reference_points_cam = torch.matmul(
                lidar2img.to(torch.float32), reference_points.unsqueeze(-1).to(torch.float32)).squeeze(-1)
            
            # Step 6: Check if points are in front of camera (z_cam > 0)
            # Points behind the camera are not visible
            eps = 1e-5  # Small epsilon to avoid numerical issues
            bev_mask = (reference_points_cam[..., 2:3] > eps)
            
            # Step 7: Perspective division to get 2D image coordinates
            # x_img = x_cam / z_cam, y_img = y_cam / z_cam
            # Use maximum to avoid division by zero (points very close to camera)
            z = torch.maximum(reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3]) * eps)
            reference_points_cam = reference_points_cam[..., 0:2] / z
            # Now reference_points_cam is [D, B, num_cam, N, 2] with (x_img, y_img) in camera coordinates
            
            # Step 8: Normalize to [0, 1] range using image dimensions
            # x_img: [0, img_width] -> [0, 1]
            # y_img: [0, img_height] -> [0, 1]
            img_shape = img_metas[0]['img_shape'][0]  # [H, W]
            reference_points_cam[..., 0] /= img_shape[1]  # Divide by width
            reference_points_cam[..., 1] /= img_shape[0]  # Divide by height
            
            # Step 9: Compute final visibility mask
            # A point is visible if:
            # - It is in front of the camera (already checked)
            # - It is within image bounds (0 < u < 1 and 0 < v < 1)
            bev_mask = (bev_mask &
                        (reference_points_cam[..., 0:1] > 0.0) & (reference_points_cam[..., 0:1] < 1.0) &
                        (reference_points_cam[..., 1:2] > 0.0) & (reference_points_cam[..., 1:2] < 1.0))
            
            # Step 10: Handle NaN values that may occur during projection
            # NaN can occur if z_cam is exactly 0 or if transformation matrices are invalid
            if digit_version(TORCH_VERSION) >= digit_version('1.8'):
                bev_mask = torch.nan_to_num(bev_mask)
            else:
                # Fallback for older PyTorch versions
                bev_mask = bev_mask.new_tensor(np.nan_to_num(bev_mask.cpu().numpy()))
            
            # Step 11: Permute to final output format
            # reference_points_cam: [D, B, num_cam, N, 2] -> [num_cam, B, N, D, 2]
            reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4)
            # bev_mask: [D, B, num_cam, N, 1] -> [num_cam, B, N, D]
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
            
            This function orchestrates the forward pass through the BEVFormer encoder:
            1. Normalizes input formats (handles both batch-first and batch-second)
            2. Generates 3D and 2D reference points
            3. Projects 3D points to camera views
            4. Builds hybrid reference points for temporal attention
            5. Processes BEV queries through multiple transformer layers
            
            Args:
                bev_query (torch.Tensor): BEV query features
                    - Shape: [num_query, bs, embed_dims] or [bs, num_query, embed_dims]
                    - num_query: H * W (total number of BEV queries)
                key (torch.Tensor): Multi-camera key features
                    - Shape: [num_cam, num_value, bs, embed_dims]
                    - num_cam: number of cameras (typically 6)
                    - num_value: total number of feature points across all levels
                value (torch.Tensor): Multi-camera value features
                    - Shape: [num_cam, num_value, bs, embed_dims]
                bev_h (int): BEV height (number of queries along Y-axis)
                bev_w (int): BEV width (number of queries along X-axis)
                bev_pos (torch.Tensor): Positional encoding for BEV queries
                    - Shape: [num_query, bs, embed_dims] or [bs, num_query, embed_dims]
                spatial_shapes (torch.Tensor): Spatial shapes of feature pyramid levels
                    - Shape: [num_levels, 2]
                    - Each row is [H_l, W_l] for level l
                level_start_index (torch.Tensor): Start indices for each level in flattened features
                    - Shape: [num_levels]
                valid_ratios (torch.Tensor, optional): Valid ratios for each level
                    - Not used in current implementation but kept for compatibility
                prev_bev (torch.Tensor, optional): Previous BEV features for temporal attention
                    - Shape: [num_query, bs, embed_dims] or [bs, num_query, embed_dims]
                    - If None, temporal attention is skipped
                shift (torch.Tensor): Shift for temporal alignment
                    - Shape: [bs, 2]
                    - Used to align current and previous BEV features
                **kwargs: Additional arguments passed to layers
                    - Must include 'img_metas': List[dict] with image metadata
                
            Returns:
                torch.Tensor: Encoded BEV features
                    - If return_intermediate=False: [bs, num_query, embed_dims]
                    - If return_intermediate=True: [num_layers, bs, num_query, embed_dims]
            """
            # Step 1: Normalize input formats to batch-first
            # Handle both [num_query, bs, embed_dims] and [bs, num_query, embed_dims] formats
            bs = bev_query.size(1) if bev_query.dim() == 3 else bev_query.size(0)
            bev_query = bev_query.permute(1, 0, 2) if bev_query.dim() == 3 else bev_query
            bev_pos = bev_pos.permute(1, 0, 2) if bev_pos.dim() == 3 else bev_pos
            
            # Step 2: Generate 3D reference points for spatial cross-attention (SCA)
            # These points are at different heights (Z anchors) for each BEV query
            # Shape: [bs, H*W, num_points_in_pillar, 3]
            ref_3d = self.get_reference_points(
                bev_h, bev_w, self.pc_range[5] - self.pc_range[2],  # Z range
                self.num_points_in_pillar, dim='3d', bs=bs,
                device=bev_query.device, dtype=bev_query.dtype)
            
            # Step 3: Generate 2D reference points for temporal self-attention (TSA)
            # These points are used to align current and previous BEV features
            # Shape: [bs, H*W, 1, 2]
            ref_2d = self.get_reference_points(
                bev_h, bev_w, dim='2d', bs=bs,
                device=bev_query.device, dtype=bev_query.dtype)
            
            # Step 4: Project 3D reference points to camera views
            # This computes where each 3D point appears in each camera image
            # reference_points_cam: [num_cam, bs, H*W, num_points_in_pillar, 2]
            # bev_mask: [num_cam, bs, H*W, num_points_in_pillar] (visibility mask)
            reference_points_cam, bev_mask = self.point_sampling(
                ref_3d, self.pc_range, kwargs['img_metas'])
            
            # Step 5: Apply shift for temporal alignment
            # The shift accounts for ego-motion between current and previous frames
            # shift_ref_2d: [bs, H*W, 1, 2] (shifted reference points for previous frame)
            shift_ref_2d = ref_2d.clone()
            shift_ref_2d += shift[:, None, None, :]  # Broadcast shift to all queries
            
            # Step 6: Build hybrid reference points for temporal attention
            # We stack current and previous BEV queries/features for temporal self-attention
            # This allows the model to attend to both current and previous BEV features
            len_bev, num_bev_level = ref_2d.shape[1], ref_2d.shape[2]  # H*W, 1
            if prev_bev is not None:
                # Normalize prev_bev format and stack with current BEV query
                prev_bev = prev_bev.permute(1, 0, 2) if prev_bev.dim() == 3 else prev_bev
                # Stack: [bs, num_query, embed_dims] -> [bs*2, num_query, embed_dims]
                # First bs items are prev_bev, next bs items are current bev_query
                prev_bev = torch.stack([prev_bev, bev_query], 1).reshape(bs * 2, len_bev, -1)
                # Stack reference points: [bs*2, num_query, 1, 2]
                # First bs items use shifted points (for prev_bev), next bs items use current points
                hybird_ref_2d = torch.stack([shift_ref_2d, ref_2d], 1).reshape(bs * 2, len_bev, num_bev_level, 2)
            else:
                # No previous BEV available, use current points for both
                hybird_ref_2d = torch.stack([ref_2d, ref_2d], 1).reshape(bs * 2, len_bev, num_bev_level, 2)
                prev_bev = None
            
            # Step 7: Process BEV queries through transformer layers
            # Each layer performs:
            # - Temporal Self-Attention (TSA): Aggregates information from previous BEV
            # - Spatial Cross-Attention (SCA): Aggregates information from multi-camera features
            # - Feed-Forward Network (FFN): Non-linear transformation
            intermediate = []
            for layer in self.layers:
                bev_query = layer(
                    bev_query, key, value, *args,
                    bev_pos=bev_pos,  # Positional encoding
                    ref_2d=hybird_ref_2d,  # 2D reference points for TSA
                    ref_3d=ref_3d,  # 3D reference points for SCA
                    bev_h=bev_h, bev_w=bev_w,  # BEV spatial dimensions
                    spatial_shapes=spatial_shapes,  # Feature pyramid shapes
                    level_start_index=level_start_index,  # Level start indices
                    reference_points_cam=reference_points_cam,  # Projected 2D points
                    bev_mask=bev_mask,  # Visibility mask
                    prev_bev=prev_bev,  # Previous BEV features
                    **kwargs)
                if self.return_intermediate:
                    intermediate.append(bev_query)
            
            # Return intermediate outputs if requested, otherwise return final output
            return torch.stack(intermediate) if self.return_intermediate else bev_query


