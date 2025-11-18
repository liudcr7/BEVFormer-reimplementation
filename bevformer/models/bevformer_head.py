from typing import Dict, List, Optional, Tuple
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner.base_module import BaseModule
from mmdet.models.builder import HEADS
from mmdet.models.utils import build_transformer
from .bbox.assigner import HungarianAssigner3D, normalize_bbox
from .bbox.coder import NMSFreeCoder
from mmdet.core.bbox import build_assigner, build_bbox_coder
from mmdet.models.builder import build_loss
from mmdet.core import reduce_mean

# Import positional encoding classes from mmdet
try:
    from mmdet.models.utils.positional_encoding import (
        LearnedPositionalEncoding,
        SinePositionalEncoding
    )
except ImportError:
    # Fallback: try to import from different location
    try:
        from mmdet.models.utils import LearnedPositionalEncoding, SinePositionalEncoding
    except ImportError:
        LearnedPositionalEncoding = None
        SinePositionalEncoding = None


def build_positional_encoding(cfg):
    """Build positional encoding from config dict.
    
    Args:
        cfg (dict): Config dict with 'type' key specifying the positional encoding type.
        
    Returns:
        Positional encoding module instance.
    """
    if cfg is None:
        return None
    
    cfg = cfg.copy()
    pos_encoding_type = cfg.pop('type')
    
    if pos_encoding_type == 'LearnedPositionalEncoding':
        if LearnedPositionalEncoding is None:
            raise ImportError('LearnedPositionalEncoding is not available. Please check mmdet installation.')
        return LearnedPositionalEncoding(**cfg)
    elif pos_encoding_type == 'SinePositionalEncoding':
        if SinePositionalEncoding is None:
            raise ImportError('SinePositionalEncoding is not available. Please check mmdet installation.')
        return SinePositionalEncoding(**cfg)
    else:
        raise ValueError(f'Unknown positional encoding type: {pos_encoding_type}')


@HEADS.register_module()
class BEVFormerHead(BaseModule):
    """BEVFormer head with encoder, temporal, spatial, and decoder."""
    
    def __init__(self, 
                 bev_h: int = 200,
                 bev_w: int = 200,
                 num_query: int = 900,
                 num_classes: int = 10,
                 in_channels: int = 256,
                 num_reg_fcs: int = 2,
                 code_size: int = 10,
                 code_weights: Optional[List[float]] = None,
                 bg_cls_weight: float = 0.1,
                 sync_cls_avg_factor: bool = True,
                 with_box_refine: bool = True,
                 transformer: Optional[dict] = None,
                 bbox_coder: Optional[dict] = None,
                 positional_encoding: Optional[dict] = None,
                 loss_cls: Optional[dict] = None,
                 loss_bbox: Optional[dict] = None,
                 assigner: Optional[dict] = None,
                 **kwargs):
        super().__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.num_query = num_query
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.code_size = code_size
        self.sync_cls_avg_factor = sync_cls_avg_factor
        self.with_box_refine = with_box_refine
        self.num_reg_fcs = num_reg_fcs
        self.bg_cls_weight = bg_cls_weight
        
        # Code weights for regression loss (default: velocity dimensions have lower weight)
        if code_weights is not None:
            self.code_weights = code_weights
        else:
            # Default weights: [cx, cy, cz, log(w), log(l), log(h), sin(rot), cos(rot), vx, vy]
            # Velocity dimensions (vx, vy) have lower weight (0.2) as they are harder to predict
            self.code_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]
        
        # Convert to Parameter for proper device/dtype handling
        self.code_weights = nn.Parameter(
            torch.tensor(self.code_weights, dtype=torch.float32, requires_grad=False),
            requires_grad=False
        )
        
        self.transformer = build_transformer(transformer)
        self.positional_encoding = build_positional_encoding(positional_encoding)
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.assigner = build_assigner(assigner)
        self.bbox_coder = build_bbox_coder(bbox_coder)


        # Real world dimensions from bbox_coder
        if bbox_coder is not None:
            pc_range = bbox_coder.pc_range
            self.pc_range = pc_range
            self.real_w = pc_range[3] - pc_range[0]
            self.real_h = pc_range[4] - pc_range[1]
        else:
            self.pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
            self.real_w = 102.4
            self.real_h = 102.4
        
        # Initialize layers (must be called after loss_cls is built)
        # This initializes all network layers and their weights in one place
        self._init_layers()
    
    def _init_layers(self):
        """Initialize classification and regression branches.
        
        This method initializes:
        1. Classification and regression branches for each decoder layer
        2. BEV and query embeddings (learnable embeddings for BEVFormer)
        3. Bias initialization for classification branches
        4. Transformer weights initialization
        """
        from mmcv.cnn import Linear
        from mmcv.cnn import bias_init_with_prob
        
        # Classification branch: FC layers with LayerNorm and ReLU
        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.in_channels, self.in_channels))
            cls_branch.append(nn.LayerNorm(self.in_channels))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(Linear(self.in_channels, self.num_classes))
        fc_cls = nn.Sequential(*cls_branch)
        
        # Regression branch: FC layers with ReLU
        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.in_channels, self.in_channels))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.in_channels, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)
        
        # Number of predictions equals number of decoder layers
        # Each decoder layer produces one set of predictions
        if hasattr(self.transformer, 'decoder') and self.transformer.decoder is not None:
            if hasattr(self.transformer.decoder, 'num_layers'):
                num_pred = self.transformer.decoder.num_layers
            elif hasattr(self.transformer.decoder, 'layers'):
                num_pred = len(self.transformer.decoder.layers)
            else:
                num_pred = 6  # Default fallback
        else:
            num_pred = 6  # Default fallback
        
        # Create branches for each prediction layer
        # If with_box_refine=True: each layer has independent branches (cloned)
        # If with_box_refine=False: all layers share the same branches (reference)
        if self.with_box_refine:
            self.cls_branches = nn.ModuleList([copy.deepcopy(fc_cls) for i in range(num_pred)])  # [num_pred] independent branches
            self.reg_branches = nn.ModuleList([copy.deepcopy(reg_branch) for i in range(num_pred)])  # [num_pred] independent branches
        else:
            self.cls_branches = nn.ModuleList([fc_cls for _ in range(num_pred)])  # Shared branches
            self.reg_branches = nn.ModuleList([reg_branch for _ in range(num_pred)])  # Shared branches
        
        # Initialize BEV queries and object queries
        # BEV embedding: [bev_h*bev_w, embed_dims] - learnable embeddings for BEV grid positions
        # Query embedding: [num_query, embed_dims*2] - learnable embeddings for object queries
        #   The *2 is because it contains both query and query_pos (positional encoding)
        self.bev_embedding = nn.Embedding(self.bev_h * self.bev_w, self.in_channels)
        self.query_embedding = nn.Embedding(self.num_query, self.in_channels * 2)
        
        # Initialize bias for classification branches (focal loss with sigmoid)
        # This ensures proper initialization for background class prediction
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)  # Initialize bias for background class
            for m in self.cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)
        
        # Initialize transformer weights if needed
        if hasattr(self.transformer, 'init_weights'):
            self.transformer.init_weights()
    
    def forward(self, mlvl_feats, img_metas, prev_bev=None, only_bev=False):
        """Forward function.
        
        Args:
            mlvl_feats: List of multi-view multi-level features
                - Each element: [B, num_cam, C_l, H_l, W_l]
                - Length: num_feature_levels (typically 4 for FPN)
                - Example: [[B, 6, 256, 64, 176], [B, 6, 256, 32, 88], ...]
            img_metas: List[dict] of image meta information
                - Each dict contains: 'lidar2img', 'can_bus', 'img_shape', etc.
            prev_bev: Previous BEV features (optional)
                - Shape: [B, bev_h*bev_w, embed_dims] or [bev_h*bev_w, B, embed_dims]
                - Used for temporal fusion in encoder
            only_bev: If True, only compute BEV features (for history frames)
                - When True: returns [B, bev_h*bev_w, embed_dims]
                - When False: returns dict with detection outputs
            
        Returns:
            If only_bev=True:
                - [B, bev_h*bev_w, embed_dims] BEV features
            If only_bev=False:
                - dict with keys:
                    - 'bev_embed': [B, bev_h*bev_w, embed_dims]
                    - 'all_cls_scores': [num_dec_layers, B, num_query, num_classes]
                    - 'all_bbox_preds': [num_dec_layers, B, num_query, code_size]
        """
        bs, num_cam, _, _, _ = mlvl_feats[0].shape  # Extract batch size and num_cameras
        dtype = mlvl_feats[0].dtype  # Preserve data type (float32/float16)
        
        # Initialize BEV queries and object queries
        # BEV queries: learnable embeddings for BEV grid positions
        object_query_embeds = self.query_embedding.weight.to(dtype)  # [num_query, embed_dims*2]
        bev_queries = self.bev_embedding.weight.to(dtype)  # [bev_h*bev_w, embed_dims]
        
        # Generate positional encoding for BEV
        # Create dummy mask for positional encoding generation
        bev_mask = torch.zeros((bs, self.bev_h, self.bev_w), device=bev_queries.device).to(dtype)
        # Generate positional encoding: [B, embed_dims, bev_h, bev_w] (standard MMDetection format)
        bev_pos = self.positional_encoding(bev_mask).to(dtype)
        
        if only_bev:
            # Only compute BEV features (for history frames)
            return self.transformer.get_bev_features(
                mlvl_feats,
                bev_queries,
                self.bev_h,
                self.bev_w,
                grid_length=(self.real_h / self.bev_h, self.real_w / self.bev_w),
                bev_pos=bev_pos,
                img_metas=img_metas,
                prev_bev=prev_bev,
            )
        else:
            # Full forward: encoder + decoder
            outputs = self.transformer(
                mlvl_feats,
                bev_queries,
                object_query_embeds,
                self.bev_h,
                self.bev_w,
                grid_length=(self.real_h / self.bev_h, self.real_w / self.bev_w),
                bev_pos=bev_pos,
                reg_branches=self.reg_branches if self.with_box_refine else None,  # For box refinement
                img_metas=img_metas,
                prev_bev=prev_bev
            )
            
            # Unpack transformer outputs
            # bev_embed: [B, bev_h*bev_w, embed_dims] - BEV features from encoder
            # hs: [num_dec_layers, num_query, B, embed_dims] - decoder hidden states (sequence-first)
            # all_references: [num_dec_layers + 1, B, num_query, 3] - reference points (first is initial, rest are per layer)
            bev_embed, hs, all_references = outputs
            
            # Process decoder outputs
            # Convert from sequence-first to batch-first: [num_dec_layers, B, num_query, embed_dims]
            hs = hs.permute(0, 2, 1, 3)  # [num_dec_layers, B, num_query, embed_dims]
            outputs_classes = []
            outputs_coords = []
            
            # Process each decoder layer output
            for lvl in range(hs.shape[0]):
                # Get reference points for this layer
                # all_references[lvl] corresponds to reference points before layer lvl
                reference = all_references[lvl]  # [B, num_query, 3] - (x, y, z) in [0, 1]
                
                # Apply inverse sigmoid to reference points
                # Convert from [0, 1] to (-inf, +inf) for addition
                from mmdet.models.utils.transformer import inverse_sigmoid
                reference = inverse_sigmoid(reference)  # [B, num_query, 3]
                
                # Classification and regression branches
                # Classification: [B, num_query, num_classes] - logits before sigmoid
                outputs_class = self.cls_branches[lvl](hs[lvl])  # [B, num_query, num_classes]
                # Regression: [B, num_query, code_size] - raw predictions
                # code_size=10: [cx, cy, cz, log(w), log(l), log(h), sin(rot), cos(rot), vx, vy]
                tmp = self.reg_branches[lvl](hs[lvl])  # [B, num_query, code_size]
                
                # Update reference points with regression predictions
                # Reference points format: [x, y, z] where x,y are BEV coordinates, z is height
                assert reference.shape[-1] == 3, f"Expected reference points with 3 dims, got {reference.shape[-1]}"
                
                # Update x, y coordinates (BEV plane)
                # tmp[..., 0:2]: [cx, cy] - add reference x,y and apply sigmoid
                tmp[..., 0:2] += reference[..., 0:2]  # Add reference offset
                tmp[..., 0:2] = tmp[..., 0:2].sigmoid()  # Normalize to [0, 1]
                # Update z coordinate (height)
                tmp[..., 4:5] += reference[..., 2:3]  # Add reference z offset
                tmp[..., 4:5] = tmp[..., 4:5].sigmoid()  # Normalize to [0, 1]

                # Convert normalized coordinates to real world coordinates
                # pc_range: [x_min, y_min, z_min, x_max, y_max, z_max]
                if hasattr(self, 'pc_range'):
                    # x coordinate: [0, 1] -> [x_min, x_max]
                    tmp[..., 0:1] = (tmp[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0])
                    # y coordinate: [0, 1] -> [y_min, y_max]
                    tmp[..., 1:2] = (tmp[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1])
                    # z coordinate: [0, 1] -> [z_min, z_max]
                    tmp[..., 4:5] = (tmp[..., 4:5] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2])
                
                outputs_classes.append(outputs_class)
                outputs_coords.append(tmp)  # Real-world coordinates (used for both output and loss)
            
            # Stack all layer outputs
            outputs_classes = torch.stack(outputs_classes)  # [num_dec_layers, B, num_query, num_classes]
            outputs_coords = torch.stack(outputs_coords)  # [num_dec_layers, B, num_query, code_size]
            
            outs = {
                'bev_embed': bev_embed,
                'all_cls_scores': outputs_classes,
                'all_bbox_preds': outputs_coords,
            }
            
            return outs
    
    def loss(self,
             gt_bboxes_3d,
             gt_labels_3d,
             outs,
             img_metas=None,
             **kwargs) -> Dict[str, torch.Tensor]:
        """Compute losses aggregated across all decoder layers.
        
        Args:
            gt_bboxes_3d: List[Tensor] ground truth 3D boxes
            gt_labels_3d: List[Tensor] ground truth labels
            outs: Dictionary containing predictions
                - 'all_cls_scores': [num_dec_layers, B, num_query, num_classes]
                - 'all_bbox_preds': [num_dec_layers, B, num_query, code_size]
        Returns:
            Dict[str, Tensor]: Loss dictionary
                - 'loss_cls': Classification loss from last decoder layer
                - 'loss_bbox': Regression loss from last decoder layer
                - 'd{i}.loss_cls': Auxiliary classification loss from layer i
                - 'd{i}.loss_bbox': Auxiliary regression loss from layer i
        
        Note: BEVFormer uses single-stage detection, so only decoder losses are computed.
        """
        all_cls_scores = outs['all_cls_scores']  # [num_dec_layers, B, num_query, num_classes]
        all_bbox_preds_norm = outs['all_bbox_preds']  # [num_dec_layers, B, num_query, code_size]

        # Handle empty ground truth case
        if gt_bboxes_3d is None or gt_labels_3d is None:
            dummy = all_cls_scores[-1].sum() * 0.0
            return {'loss_cls': dummy, 'loss_bbox': dummy}

        # Convert gt_bboxes_3d from LiDARInstance3DBoxes to tensor format if needed
        device = gt_labels_3d[0].device if len(gt_labels_3d) > 0 and isinstance(gt_labels_3d[0], torch.Tensor) else all_cls_scores.device
        
        # Check if gt_bboxes_3d contains LiDARInstance3DBoxes objects
        try:
            from mmdet3d.core.bbox import BaseInstance3DBoxes
            if len(gt_bboxes_3d) > 0 and isinstance(gt_bboxes_3d[0], BaseInstance3DBoxes):
                # Convert from LiDARInstance3DBoxes to tensor: [cx, cy, cz, w, l, h, rot, vx, vy, ...]
                gt_bboxes_3d = [
                    torch.cat((gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]), dim=1).to(device)
                    for gt_bboxes in gt_bboxes_3d
                ]
        except ImportError:
            # mmdet3d not available, assume gt_bboxes_3d is already tensor format
            pass

        num_layers = all_cls_scores.shape[0]  # Number of decoder layers
        loss_dict: Dict[str, torch.Tensor] = {}

        # Compute losses for each decoder layer
        for layer in range(num_layers):
            cls_scores = all_cls_scores[layer]  # [B, num_query, num_classes]
            bbox_preds_norm = all_bbox_preds_norm[layer]  # [B, num_query, code_size]
            loss_cls_layer, loss_bbox_layer = self._loss_per_layer(
                cls_scores, bbox_preds_norm, gt_bboxes_3d, gt_labels_3d
            )
            # Last layer losses are the main losses
            if layer == num_layers - 1:
                loss_dict['loss_cls'] = loss_cls_layer
                loss_dict['loss_bbox'] = loss_bbox_layer
            else:
                # Auxiliary losses from intermediate layers
                loss_dict[f'd{layer}.loss_cls'] = loss_cls_layer
                loss_dict[f'd{layer}.loss_bbox'] = loss_bbox_layer

        return loss_dict

    def _loss_per_layer(self,
                        cls_scores: torch.Tensor,
                        bbox_preds_norm: torch.Tensor,
                        gt_bboxes_3d,
                        gt_labels_3d) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute classification and regression loss for a single decoder layer.
        
        Args:
            cls_scores: [B, num_query, num_classes] classification logits
            bbox_preds_norm: [B, num_query, code_size] bbox predictions in real-world coordinates
                - Format: [cx, cy, cz, log(w), log(l), log(h), sin(rot), cos(rot), vx, vy]
                - cx, cy, cz are in real-world coordinates (not normalized to [0, 1])
            gt_bboxes_3d: List[Tensor] ground truth 3D boxes
                - Each tensor: [num_gt, code_size] in real-world coordinates
            gt_labels_3d: List[Tensor] ground truth labels
                - Each tensor: [num_gt] class indices
        
        Returns:
            loss_cls: Classification loss (scalar tensor)
            loss_bbox: Regression loss (scalar tensor)
        """
        B, Q, C = cls_scores.shape  # B=batch_size, Q=num_query, C=num_classes
        
        # Process each sample to get targets and matched indices
        labels_list = []
        label_weights_list = []
        bbox_targets_list = []
        bbox_weights_list = []
        num_total_pos = 0
        num_total_neg = 0
        
        device = cls_scores.device
        
        # Process each sample in batch
        for b in range(B):
            cls_b = cls_scores[b]  # [num_query, num_classes]
            bbox_b = bbox_preds_norm[b]  # [num_query, code_size]
            gtb = gt_bboxes_3d[b]  # [num_gt, code_size] or empty tensor
            gtl = gt_labels_3d[b]  # [num_gt] or empty tensor

            if not isinstance(gtb, torch.Tensor):
                gtb = torch.tensor(gtb, device=device, dtype=bbox_b.dtype)
            else:
                gtb = gtb.to(device).to(bbox_b.dtype)
            if not isinstance(gtl, torch.Tensor):
                gtl = torch.tensor(gtl, device=device, dtype=torch.long)
            else:
                gtl = gtl.to(device).long()

            num_gt = gtb.shape[0]  # Number of ground truth boxes for this sample
            
            # Hungarian matching: assign predictions to ground truth
            assign_result = self.assigner.assign(bbox_b, cls_b, gtb, gtl)
            pos_mask = assign_result.gt_inds > 0  # [num_query] - True for positive matches
            neg_mask = assign_result.gt_inds == 0  # [num_query] - True for negative matches
            
            pos_inds = pos_mask.nonzero(as_tuple=False).squeeze(1)  # [num_pos]
            neg_inds = neg_mask.nonzero(as_tuple=False).squeeze(1)  # [num_neg]
            
            num_total_pos += pos_inds.numel()
            num_total_neg += neg_inds.numel()
            
            # Create labels: [num_query] - class indices (num_classes for background)
            labels = torch.full((Q,), self.num_classes, dtype=torch.long, device=device)
            if pos_inds.numel() > 0:
                gt_inds = assign_result.gt_inds[pos_inds] - 1  # [num_pos] - GT indices (0-indexed)
                labels[pos_inds] = gtl[gt_inds]
            
            # Label weights: all ones
            label_weights = torch.ones(Q, dtype=torch.float32, device=device)
            
            # BBox targets and weights
            # This matches the original BEVFormer implementation where bbox_targets stores raw GT boxes
            if num_gt > 0:
                bbox_targets = torch.zeros(Q, gtb.size(-1), dtype=bbox_b.dtype, device=device)
                # bbox_weights should have shape [num_query, code_size] to match bbox_pred format
                bbox_weights = torch.zeros(Q, self.code_size, dtype=bbox_b.dtype, device=device)  # [num_query, code_size]
                
                if pos_inds.numel() > 0:
                    bbox_targets[pos_inds] = gtb[gt_inds]  # Use raw GT boxes (not normalized)
                    bbox_weights[pos_inds] = 1.0
            else:
                bbox_targets = torch.zeros(Q, bbox_preds_norm.size(-1), dtype=bbox_b.dtype, device=device)
                bbox_weights = torch.zeros(Q, self.code_size, dtype=bbox_b.dtype, device=device)  # [num_query, code_size]
            
            labels_list.append(labels)
            label_weights_list.append(label_weights)
            bbox_targets_list.append(bbox_targets)
            bbox_weights_list.append(bbox_weights)
        
        # Concatenate all samples
        labels = torch.cat(labels_list, 0)  # [B*num_query]
        label_weights = torch.cat(label_weights_list, 0)  # [B*num_query]
        bbox_targets = torch.cat(bbox_targets_list, 0)  # [B*num_query, code_size]
        bbox_weights = torch.cat(bbox_weights_list, 0)  # [B*num_query, code_size]
        
        # Reshape predictions for loss computation
        cls_scores_flat = cls_scores.reshape(-1, C)  # [B*num_query, num_classes]
        bbox_preds_flat = bbox_preds_norm.reshape(-1, bbox_preds_norm.size(-1))  # [B*num_query, code_size]
        
        # Classification loss with avg_factor (matching DETR style)
        # Construct weighted avg_factor: num_pos + num_neg * bg_cls_weight
        cls_avg_factor = num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(cls_scores_flat.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)
        
        # Use self.loss_cls (MMDetection FocalLoss)
        # self.loss_cls expects:
        #   - cls_scores: [B*num_query, num_classes] - classification logits
        #   - labels: [B*num_query] - class indices (num_classes for background)
        #   - label_weights: [B*num_query] - label weights
        #   - avg_factor: scalar - average factor for normalization
        loss_cls = self.loss_cls(
            cls_scores_flat, labels, label_weights, avg_factor=cls_avg_factor
        )
        
        # Regression loss with code_weights
        # Compute average number of positive samples across all GPUs (for distributed training)
        num_total_pos_tensor = loss_cls.new_tensor([num_total_pos])
        num_total_pos_synced = torch.clamp(reduce_mean(num_total_pos_tensor), min=1).item()
        
        # Normalize bbox targets
        normalized_bbox_targets = normalize_bbox(bbox_targets, getattr(self, 'pc_range', None))
        
        # Filter out invalid targets (NaN or Inf)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)  # [B*num_query]
        
        # Apply code_weights to bbox_weights
        # bbox_weights already has shape [B*num_query, code_size] from concatenation
        bbox_weights = bbox_weights * self.code_weights  # [B*num_query, code_size]
        
        # self.loss_bbox expects:
        #   - bbox_preds: [num_valid, code_size] - predictions (only first 10 dims used)
        #   - bbox_targets: [num_valid, code_size] - normalized targets (only first 10 dims used)
        #   - bbox_weights: [num_valid, code_size] - weights with code_weights applied (only first 10 dims used)
        #   - avg_factor: scalar - average factor for normalization
        loss_bbox = self.loss_bbox(
            bbox_preds_flat[isnotnan, :10], 
            normalized_bbox_targets[isnotnan, :10], 
            bbox_weights[isnotnan, :10],
            avg_factor=num_total_pos_synced
        )
        
        # Handle NaN values (for PyTorch >= 1.8)
        try:
            if hasattr(torch, 'nan_to_num') and torch.__version__ >= '1.8.0':
                loss_cls = torch.nan_to_num(loss_cls)
                loss_bbox = torch.nan_to_num(loss_bbox)
        except:
            pass

        return loss_cls, loss_bbox
    
    def get_bboxes(self, 
                   outs: Dict[str, torch.Tensor], 
                   img_metas: List[dict] = None,
                   score_thr: float = 0.3, 
                   **kwargs):
        """Get bounding boxes from predictions.
        
        Args:
            outs: Dict with 'all_cls_scores' and 'all_bbox_preds'
            img_metas: List of image meta information
            score_thr: Score threshold for filtering
            
        Returns:
            List of [bboxes, scores, labels] tuples matching original BEVFormer format
        """
        preds_dicts = {
            'all_cls_scores': outs['all_cls_scores'],
            'all_bbox_preds': outs['all_bbox_preds']
        }

        decoded = self.bbox_coder.decode(preds_dicts)
        
        num_samples = len(decoded)
        ret_list = []
        for i in range(num_samples):
            preds = decoded[i]
            bboxes = preds['bboxes']
            
            # Convert center height to bottom height (matching original BEVFormer)
            # bboxes format: [cx, cy, cz, w, l, h, rot, vx, vy, ...]
            # cz is center height, we need bottom height = cz - h/2
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
            
            # Convert to specific 3D box type (e.g., LiDARInstance3DBoxes)
            code_size = bboxes.shape[-1]
            if img_metas is not None and i < len(img_metas) and 'box_type_3d' in img_metas[i]:
                bboxes = img_metas[i]['box_type_3d'](bboxes, code_size)
            
            scores = preds['scores']
            labels = preds['labels']
            
            # Return format matching original BEVFormer: [bboxes, scores, labels]
            ret_list.append([bboxes, scores, labels])
        
        return ret_list
