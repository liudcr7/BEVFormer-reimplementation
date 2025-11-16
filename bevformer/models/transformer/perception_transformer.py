"""PerceptionTransformer for BEVFormer."""
from typing import Optional, List
import numpy as np
import torch
import torch.nn as nn
from mmdet.models.utils.builder import TRANSFORMER
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmcv.cnn import xavier_init
from mmcv.runner.base_module import BaseModule
from mmcv.runner import auto_fp16
from torch.nn.init import normal_
from torchvision.transforms.functional import rotate


@TRANSFORMER.register_module()
class PerceptionTransformer(BaseModule):
    """PerceptionTransformer coordinates BEV feature extraction and object detection.
    
    This module integrates:
    - Multi-view image feature aggregation via spatial cross-attention
    - Temporal fusion via self-attention on BEV features
    - Object detection via transformer decoder
    
    The implementation processes multi-view camera features to generate BEV (Bird's Eye View)
    representations and performs object detection in 3D space.
    """
    
    def __init__(
        self,
        embed_dims: int = 256,
        num_feature_levels: int = 4,
        num_cams: int = 6,
        encoder: Optional[dict] = None,
        decoder: Optional[dict] = None,
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
        
        self.encoder = build_transformer_layer_sequence(encoder) if encoder else None
        self.decoder = build_transformer_layer_sequence(decoder) if decoder else None
        
        self._init_embeddings()
    
    def _init_embeddings(self):
        """Initialize learnable embeddings and projection layers.
        
        This method creates:
        - level_embeds: Embeddings for different feature pyramid levels
        - cams_embeds: Embeddings for different camera views
        - reference_points: Linear layer to generate initial reference points for decoder
        - can_bus_mlp: MLP to process CAN bus signals for ego motion compensation
        """
        # Level embeddings for multi-scale features
        # Shape: [num_feature_levels, embed_dims] - typically [4, 256]
        self.level_embeds = nn.Parameter(torch.Tensor(self.num_feature_levels, self.embed_dims))
        
        # Camera embeddings for multi-view features
        # Shape: [num_cams, embed_dims] - typically [6, 256]
        self.cams_embeds = nn.Parameter(torch.Tensor(self.num_cams, self.embed_dims))
        
        # Reference points generator for decoder
        # Input: [B, num_query, embed_dims] -> Output: [B, num_query, 3] (x, y, z)
        self.reference_points = nn.Linear(self.embed_dims, 3)
        
        # CAN bus MLP for ego motion compensation
        # Input: [B, 18] CAN bus vector -> Output: [B, embed_dims]
        layers = [
            nn.Linear(18, self.embed_dims // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims // 2, self.embed_dims),
            nn.ReLU(inplace=True),
        ]
        if self.can_bus_norm:
            layers.append(nn.LayerNorm(self.embed_dims))
        self.can_bus_mlp = nn.Sequential(*layers)
    
    def init_weights(self):
        """Initialize transformer weights.
        
        This method:
        1. Initializes all 2D+ parameters with xavier uniform
        2. Calls init_weights() on all submodules (attention modules, etc.)
        3. Initializes embeddings with normal distribution
        4. Initializes linear layers with xavier uniform
        """
        # Initialize all parameters with xavier uniform (for 2D+ tensors)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
        # Initialize attention modules (SpatialCrossAttention, TemporalSelfAttention, etc.)
        for m in self.modules():
            if hasattr(m, 'init_weights'):
                m.init_weights()
        
        # Initialize embeddings with normal distribution
        normal_(self.level_embeds)
        normal_(self.cams_embeds)
        
        # Initialize linear layers with xavier uniform
        xavier_init(self.reference_points, distribution='uniform', bias=0.)
        xavier_init(self.can_bus_mlp, distribution='uniform', bias=0.)
    
    def _compute_temporal_shift(self, img_metas: List[dict], grid_length: List[float], 
                                bev_h: int, bev_w: int, device: torch.device) -> torch.Tensor:
        """Compute temporal shift for ego motion compensation.
        
        This function computes the shift in BEV grid coordinates based on ego motion
        from CAN bus signals. The shift accounts for both translation and rotation
        of the ego vehicle between frames.
        
        Args:
            img_metas: List of image meta information, each containing 'can_bus' key
                - can_bus[0]: x translation (meters)
                - can_bus[1]: y translation (meters)
                - can_bus[-2]: yaw angle (radians)
            grid_length: [grid_length_y, grid_length_x] grid size in meters
            bev_h, bev_w: BEV spatial dimensions (number of grid cells)
            device: Device to create tensors on
            
        Returns:
            shift: [B, 2] tensor with (shift_x, shift_y) in BEV grid coordinates
                - shift_x: shift in x direction (columns) normalized by grid width
                - shift_y: shift in y direction (rows) normalized by grid height
                - Values are in [0, 1] range representing fraction of grid cell
        """
        # Extract CAN bus data: [dx, dy, yaw] for each sample
        # can_bus format: [x, y, z, roll, pitch, yaw, vx, vy, vz, ax, ay, az, ...]
        can_bus_data = np.array([[m['can_bus'][0], m['can_bus'][1], m['can_bus'][-2]] 
                                  for m in img_metas])  # [B, 3]: [dx, dy, yaw]
        
        # Extract translation deltas and ego rotation angle
        delta_xy = can_bus_data[:, :2]  # [B, 2] - translation in x and y directions
        ego_angle = can_bus_data[:, 2] / np.pi * 180  # [B] - yaw angle converted to degrees
        
        # Compute translation magnitude and direction
        translation_length = np.linalg.norm(delta_xy, axis=1)  # [B] - magnitude of translation
        translation_angle = np.arctan2(delta_xy[:, 1], delta_xy[:, 0]) / np.pi * 180  # [B] - direction angle (degrees)
        
        # Compute BEV angle: difference between ego rotation and translation direction
        # This accounts for the fact that the vehicle may be rotating while translating
        bev_angle = ego_angle - translation_angle  # [B] - angle in BEV coordinate system
        
        # Convert angle to radians for trigonometric functions
        angle_rad = np.deg2rad(bev_angle)  # [B]
        
        # Compute shift in BEV grid coordinates
        # shift_y: shift in y direction (rows) normalized by grid height
        shift_y = translation_length * np.cos(angle_rad) / grid_length[0] / bev_h  # [B]
        # shift_x: shift in x direction (cols) normalized by grid width
        shift_x = translation_length * np.sin(angle_rad) / grid_length[1] / bev_w  # [B]
        
        # Apply shift flag (can be disabled for ablation studies)
        if self.use_shift:
            shift = torch.tensor([shift_x, shift_y], device=device, dtype=torch.float32).T  # [B, 2]
        else:
            shift = torch.zeros(len(img_metas), 2, device=device, dtype=torch.float32)
        
        return shift
    
    def _rotate_bev_features(self, prev_bev: torch.Tensor, bev_h: int, bev_w: int,
                             img_metas: List[dict]) -> torch.Tensor:
        """Rotate previous BEV features to align with current frame.
        
        This function rotates each sample's previous BEV features according to the
        ego vehicle's rotation (yaw angle) to align with the current frame's orientation.
        This is crucial for temporal fusion when the vehicle rotates between frames.
        
        Args:
            prev_bev: [bev_h*bev_w, B, embed_dims] previous BEV features in sequence-first format
            bev_h, bev_w: BEV spatial dimensions (number of grid cells)
            img_metas: List of image meta information, each containing 'can_bus' key
                - can_bus[-1]: yaw angle in radians (ego rotation)
            
        Returns:
            Rotated prev_bev with same shape [bev_h*bev_w, B, embed_dims]
        """
        bs = prev_bev.size(1)
        for i in range(bs):
            # Get rotation angle from CAN bus (yaw angle in radians)
            rotation_angle = img_metas[i]['can_bus'][-1]  # Scalar
            
            # Reshape to spatial format: [bev_h*bev_w, embed_dims] -> [embed_dims, bev_h, bev_w]
            # This is required by the rotate function which expects [C, H, W] format
            bev_feat = prev_bev[:, i].reshape(bev_h, bev_w, -1).permute(2, 0, 1)  # [C, H, W]
            
            # Rotate BEV features around the center point
            # rotate function performs 2D rotation in image space
            bev_feat = rotate(bev_feat, rotation_angle, center=self.rotate_center)
            
            # Reshape back to sequence format: [embed_dims, bev_h, bev_w] -> [bev_h*bev_w, embed_dims]
            prev_bev[:, i] = bev_feat.permute(1, 2, 0).reshape(bev_h * bev_w, -1)
        
        return prev_bev
    
    def _prepare_multi_scale_features(self, mlvl_feats: List[torch.Tensor]) -> tuple:
        """Prepare multi-scale features with embeddings.
        
        This function processes multi-view multi-scale features by:
        1. Flattening spatial dimensions for each level
        2. Adding camera embeddings to distinguish different camera views
        3. Adding level embeddings to distinguish different FPN levels
        4. Concatenating all levels into a single tensor
        5. Computing spatial shapes and level start indices for deformable attention
        
        Args:
            mlvl_feats: List of multi-view multi-scale features
                - Each element: [B, num_cam, C, H_l, W_l]
                - Length: num_feature_levels (typically 4 for FPN)
                - Example: [[B, 6, 256, 64, 176], [B, 6, 256, 32, 88], ...]
        
        Returns:
            tuple containing:
            - feat_flatten: [num_cam, sum(H*W), B, embed_dims] concatenated features
                - All feature levels concatenated along spatial dimension
                - Format: sequence-first for transformer
            - spatial_shapes: [num_levels, 2] spatial dimensions
                - Each row: [H_l, W_l] for level l
            - level_start_index: [num_levels] starting indices for each level
                - Used by deformable attention to locate features at each level
                - Format: [0, H0*W0, H0*W0+H1*W1, ...]
        """
        feat_list = []
        spatial_shapes = []
        
        for lvl, feat in enumerate(mlvl_feats):
            bs, num_cam, c, h, w = feat.shape
            
            # Flatten spatial dimensions: [B, num_cam, C, H_l, W_l] -> [num_cam, B, H_l*W_l, C]
            feat = feat.flatten(3).permute(1, 0, 3, 2)  # [num_cam, B, H*W, C]
            
            # Add camera embeddings: [num_cam, embed_dims] -> [num_cam, 1, 1, embed_dims]
            # This allows the model to distinguish features from different camera views
            if self.use_cams_embeds:
                feat = feat + self.cams_embeds[:, None, None, :].to(feat.dtype)
            
            # Add level embeddings: [num_levels, embed_dims] -> [1, 1, 1, embed_dims]
            # This allows the model to distinguish features from different FPN levels
            feat = feat + self.level_embeds[None, None, lvl:lvl+1, :].to(feat.dtype)
            
            spatial_shapes.append((h, w))
            feat_list.append(feat)  # [num_cam, B, H*W, C]
        
        # Concatenate all levels: [num_cam, B, sum(H*W), C]
        feat_flatten = torch.cat(feat_list, dim=2)  # [num_cam, B, sum(H*W), C]
        
        # Compute spatial shapes tensor: [num_levels, 2]
        spatial_shapes = torch.tensor(spatial_shapes, dtype=torch.long, device=feat.device)
        
        # Compute level start indices: [num_levels]
        # level_start_index[l] = sum of H*W for all levels before l
        level_start_index = torch.cat([
            spatial_shapes.new_zeros(1),  # Start at 0
            spatial_shapes.prod(1).cumsum(0)[:-1]  # Cumulative sum of H*W
        ])
        
        # Permute to sequence-first format: [num_cam, sum(H*W), B, embed_dims]
        return feat_flatten.permute(0, 2, 1, 3), spatial_shapes, level_start_index
    
    @auto_fp16(apply_to=('mlvl_feats', 'bev_queries', 'prev_bev', 'bev_pos'))
    def get_bev_features(
        self,
        mlvl_feats: List[torch.Tensor],
        bev_queries: torch.Tensor,
        bev_h: int,
        bev_w: int,
        grid_length: List[float] = [0.512, 0.512],
        bev_pos: Optional[torch.Tensor] = None,
        prev_bev: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """Extract BEV features from multi-view image features.
        
        This is the core function that aggregates multi-view camera features into
        a unified BEV representation. It handles:
        1. Ego motion compensation via temporal shift
        2. Temporal fusion via previous BEV rotation
        3. Multi-view feature aggregation via spatial cross-attention
        4. Multi-scale feature processing with embeddings
        
        Args:
            mlvl_feats: List of multi-view multi-scale features
                - Each element: [B, num_cam, C, H_l, W_l]
                - Length: num_feature_levels (typically 4 for FPN)
                - Example: [[B, 6, 256, 64, 176], [B, 6, 256, 32, 88], ...]
            bev_queries: BEV query embeddings
                - Shape: [bev_h*bev_w, embed_dims] - learnable embeddings for BEV grid
                - These queries are used to aggregate features from camera views
            bev_h, bev_w: BEV spatial dimensions (e.g., 200, 200)
                - Number of grid cells in height and width directions
            grid_length: Grid size in meters [grid_length_y, grid_length_x]
                - Used to compute shift for temporal alignment
                - grid_length_y: size of each grid cell in y direction (meters)
                - grid_length_x: size of each grid cell in x direction (meters)
            bev_pos: Positional encoding for BEV queries
                - Shape: [B, embed_dims, bev_h, bev_w] (spatial format)
                - Provides spatial position information to BEV queries
            prev_bev: Previous BEV features (optional)
                - Shape: [num_query, B, embed_dims] or [B, num_query, embed_dims]
                - Used for temporal fusion across frames
                - If provided, will be rotated and fused with current BEV
            
        Returns:
            bev_embed: BEV features from encoder
                - Shape: [B, bev_h*bev_w, embed_dims]
                - Batch-first format for downstream processing
        """
        bs = mlvl_feats[0].size(0)  # Batch size
        device = bev_queries.device
        
        # Expand BEV queries to batch dimension
        # bev_queries: [bev_h*bev_w, embed_dims] -> [bev_h*bev_w, B, embed_dims]
        bev_queries = bev_queries.unsqueeze(1).repeat(1, bs, 1)  # [bev_h*bev_w, B, embed_dims]
        
        # Reshape positional encoding from spatial to sequence format
        # bev_pos: [B, embed_dims, bev_h, bev_w] -> [bev_h*bev_w, B, embed_dims]
        bev_pos = bev_pos.flatten(2).permute(2, 0, 1)  # [bev_h*bev_w, B, embed_dims]
        
        # Compute temporal shift for ego motion compensation
        # This shift accounts for vehicle translation between frames
        shift = self._compute_temporal_shift(kwargs['img_metas'], grid_length, bev_h, bev_w, device)
        
        # Rotate previous BEV if needed to align with current frame
        # This accounts for vehicle rotation between frames
        if prev_bev is not None:
            # Ensure prev_bev is in sequence-first format: [bev_h*bev_w, B, embed_dims]
            if prev_bev.dim() == 3 and prev_bev.size(1) == bev_h * bev_w:
                prev_bev = prev_bev.permute(1, 0, 2)  # [bev_h*bev_w, B, embed_dims]
            
            if self.rotate_prev_bev:
                prev_bev = self._rotate_bev_features(prev_bev, bev_h, bev_w, kwargs['img_metas'])
        
        # Add CAN bus signals to BEV queries
        # CAN bus provides ego motion information (translation, rotation, velocity, etc.)
        # This information is encoded and added to queries to help the model understand
        # the relationship between current and previous frames
        if self.use_can_bus:
            can_bus = torch.tensor([m['can_bus'] for m in kwargs['img_metas']], device=device)  # [B, 18]
            can_bus_embed = self.can_bus_mlp(can_bus)[None, :, :]  # [1, B, embed_dims] - add sequence dimension
            bev_queries = bev_queries + can_bus_embed
        
        # Prepare multi-scale features with embeddings
        # This adds camera and level embeddings, and concatenates all levels
        feat_flatten, spatial_shapes, level_start_index = self._prepare_multi_scale_features(mlvl_feats)
        
        # Encode BEV features using transformer encoder
        # The encoder performs:
        # 1. Temporal self-attention: fuses prev_bev and current BEV
        # 2. Spatial cross-attention: aggregates features from multi-view images
        bev_embed = self.encoder(
            bev_queries,  # [bev_h*bev_w, B, embed_dims]
            feat_flatten,  # [num_cam, sum(H*W), B, embed_dims] - key
            feat_flatten,  # [num_cam, sum(H*W), B, embed_dims] - value
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
        # Encoder returns sequence-first format, but we need batch-first for downstream
        return bev_embed.permute(1, 0, 2) if bev_embed.dim() == 3 else bev_embed
    
    @auto_fp16(apply_to=('mlvl_feats', 'bev_queries', 'object_query_embed', 'prev_bev', 'bev_pos'))
    def forward(
        self,
        mlvl_feats: List[torch.Tensor],
        bev_queries: torch.Tensor,
        object_query_embed: torch.Tensor,
        bev_h: int,
        bev_w: int,
        grid_length: List[float] = [0.512, 0.512],
        bev_pos: Optional[torch.Tensor] = None,
        reg_branches: Optional[nn.ModuleList] = None,
        prev_bev: Optional[torch.Tensor] = None,
        **kwargs
    ) -> tuple:
        """Forward pass through encoder and decoder.
        
        This function performs the complete forward pass:
        1. Extracts BEV features from multi-view images via encoder
        2. Generates object queries and initial reference points
        3. Decodes object queries using BEV features via decoder
        4. Returns BEV features and decoder outputs
        
        Args:
            mlvl_feats: List of multi-view multi-scale features
                - Each element: [B, num_cam, C, H_l, W_l]
                - Length: num_feature_levels (typically 4 for FPN)
            bev_queries: BEV query embeddings
                - Shape: [bev_h*bev_w, embed_dims] - learnable embeddings for BEV grid
            object_query_embed: Object query embeddings for decoder
                - Shape: [num_query, embed_dims*2]
                - Contains both query and query_pos concatenated
                - num_query: number of object queries (typically 900)
            bev_h, bev_w: BEV spatial dimensions (e.g., 200, 200)
            grid_length: Grid size in meters [grid_length_y, grid_length_x]
            bev_pos: Positional encoding for BEV queries
                - Shape: [B, embed_dims, bev_h, bev_w] (spatial format)
            reg_branches: Regression heads for box refinement (optional)
                - Type: nn.ModuleList of regression heads
                - Only passed when with_box_refine=True
                - Each head: [B, num_query, embed_dims] -> [B, num_query, code_size]
            prev_bev: Previous BEV features (optional)
                - Shape: [B, bev_h*bev_w, embed_dims] or [bev_h*bev_w, B, embed_dims]
                - Used for temporal fusion
            
        Returns:
            tuple containing:
            - bev_embed: [bev_h*bev_w, B, embed_dims] BEV features (sequence-first format)
                - Used as value in decoder cross-attention
            - inter_states: [num_dec_layers, num_query, B, embed_dims] decoder outputs
                - Hidden states from all decoder layers (sequence-first format)
                - Used to generate classification and regression predictions
            - init_reference: [B, num_query, 3] initial reference points
                - Format: (x, y, z) normalized to [0, 1] range
                - Generated from object queries before decoder
            - inter_references: [num_dec_layers, B, num_query, 3] reference points
                - Reference points at each decoder layer (if with_box_refine)
                - Updated iteratively by reg_branches
        """
        # Extract BEV features from multi-view images
        # This performs temporal fusion and spatial cross-attention
        bev_embed = self.get_bev_features(
            mlvl_feats, bev_queries, bev_h, bev_w,
            grid_length=grid_length, bev_pos=bev_pos, prev_bev=prev_bev, **kwargs
        )  # [B, bev_h*bev_w, embed_dims]
        
        # If no decoder, just return BEV features
        if self.decoder is None:
            return bev_embed.permute(1, 0, 2), None, None, None
        
        # Prepare object queries
        # object_query_embed contains both query and query_pos concatenated
        bs = mlvl_feats[0].size(0)
        query_pos, query = torch.split(object_query_embed, self.embed_dims, dim=1)
        # Expand to batch dimension: [num_query, embed_dims] -> [B, num_query, embed_dims]
        query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)  # [B, num_query, embed_dims]
        query = query.unsqueeze(0).expand(bs, -1, -1)  # [B, num_query, embed_dims]
        
        # Generate initial reference points
        # Reference points are 3D coordinates (x, y, z) normalized to [0, 1]
        # They are used to guide the decoder's attention and regression
        init_reference = self.reference_points(query_pos).sigmoid()  # [B, num_query, 3]
        
        # Convert to sequence-first format for decoder
        # Decoder expects sequence-first format: [num_query, B, embed_dims]
        query = query.permute(1, 0, 2)  # [num_query, B, embed_dims]
        query_pos = query_pos.permute(1, 0, 2)  # [num_query, B, embed_dims]
        bev_embed = bev_embed.permute(1, 0, 2)  # [bev_h*bev_w, B, embed_dims]
        
        # Decode object queries using BEV features
        # The decoder performs:
        # 1. Self-attention between object queries
        # 2. Cross-attention with BEV features
        # 3. Iterative refinement of reference points (if with_box_refine)
        inter_states, inter_references = self.decoder(
            query=query,  # [num_query, B, embed_dims]
            key=None,  # Not used in decoder
            value=bev_embed,  # [bev_h*bev_w, B, embed_dims] - BEV features as values
            query_pos=query_pos,  # [num_query, B, embed_dims]
            reference_points=init_reference,  # [B, num_query, 3]
            reg_branches=reg_branches,  # Optional: for box refinement
            spatial_shapes=torch.tensor([[bev_h, bev_w]], device=query.device, dtype=torch.long),
            level_start_index=torch.tensor([0], device=query.device, dtype=torch.long),
            **kwargs
        )
        
        return bev_embed, inter_states, init_reference, inter_references
