from typing import Dict, List, Optional, Tuple
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.builder import HEADS
from mmdet.models.builder import build_transformer
from .decoder.detr_decoder import DetectionTransformerDecoder
from .bbox.assigner import HungarianAssigner3D, normalize_bbox
from .bbox.coder import NMSFreeCoder
from .utils.focal import sigmoid_focal_loss

try:
    from mmdet.models.builder import POSITIONAL_ENCODING
except Exception:
    try:
        from mmdet.models.utils.builder import POSITIONAL_ENCODING
    except Exception:
        POSITIONAL_ENCODING = None


@POSITIONAL_ENCODING.register_module()
class LearnedPositionalEncoding(nn.Module):
    """Learned positional encoding for BEV queries."""
    def __init__(self, num_feats: int, row_num_embed: int, col_num_embed: int):
        super().__init__()
        self.row_embed = nn.Embedding(row_num_embed, num_feats)
        self.col_embed = nn.Embedding(col_num_embed, num_feats)
        self.num_feats = num_feats
        
    def forward(self, bev_h: int, bev_w: int, device: torch.device) -> torch.Tensor:
        """Generate positional encoding for BEV grid.
        
        Args:
            bev_h: Height of BEV grid
            bev_w: Width of BEV grid
            device: Device to create tensors on
            
        Returns:
            pos: [1, bev_h*bev_w, num_feats*2] positional encoding
        """
        h_embed = torch.arange(bev_h, device=device).unsqueeze(1).repeat(1, bev_w)
        w_embed = torch.arange(bev_w, device=device).unsqueeze(0).repeat(bev_h, 1)
        
        h_pos = self.row_embed(h_embed.flatten())  # [H*W, num_feats]
        w_pos = self.col_embed(w_embed.flatten())  # [H*W, num_feats]
        
        pos = torch.cat([h_pos, w_pos], dim=-1)  # [H*W, num_feats*2]
        return pos.unsqueeze(0)  # [1, H*W, num_feats*2]


@HEADS.register_module()
class BEVFormerHead(nn.Module):
    """BEVFormer head with encoder, temporal, spatial, and decoder."""
    
    def __init__(self, 
                 bev_h: int = 200,
                 bev_w: int = 200,
                 num_query: int = 900,
                 num_classes: int = 10,
                 in_channels: int = 256,
                 sync_cls_avg_factor: bool = True,
                 with_box_refine: bool = True,
                 as_two_stage: bool = False,
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
        self.sync_cls_avg_factor = sync_cls_avg_factor
        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        
        # Build transformer (PerceptionTransformer) from config
        # Use build_transformer to construct PerceptionTransformer
        self.transformer = build_transformer(transformer)

        
        # Positional encoding
        if positional_encoding is not None:
            pos_cfg = positional_encoding
            num_feats = pos_cfg.get('num_feats', in_channels // 2)
            row_num_embed = pos_cfg.get('row_num_embed', bev_h)
            col_num_embed = pos_cfg.get('col_num_embed', bev_w)
            # Try to build from registry first
            try:
                from mmdet.models.builder import build_positional_encoding
                self.positional_encoding = build_positional_encoding(positional_encoding)
            except Exception:
                # Fallback to direct instantiation
                self.positional_encoding = LearnedPositionalEncoding(
                    num_feats, row_num_embed, col_num_embed
                )
        else:
            self.positional_encoding = LearnedPositionalEncoding(
                in_channels // 2, bev_h, bev_w
            )
        
        # Decoder is now part of PerceptionTransformer, but we need to extract num_levels for reference points
        # This will be set in _init_layers based on transformer config
        
        # Query embeddings are now in _init_layers
        
        # Loss functions
        self.loss_cls_cfg = loss_cls or dict(type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=2.0)
        self.loss_bbox_cfg = loss_bbox or dict(type='L1Loss', loss_weight=0.25)
        
        # Build loss functions
        try:
            from mmdet.models.builder import build_loss
            self.loss_cls = build_loss(self.loss_cls_cfg) if loss_cls is not None else None
            self.loss_bbox = build_loss(self.loss_bbox_cfg) if loss_bbox is not None else None
        except Exception:
            # Fallback: loss functions will be computed manually
            self.loss_cls = None
            self.loss_bbox = None
        
        # Assigner
        try:
            from mmdet.models.builder import build_assigner
            self.assigner = build_assigner(assigner)
        except Exception:
            self.assigner = HungarianAssigner3D(**assigner)

        
        # BBox coder
        try:
            from mmdet.models.builder import build_bbox_coder
            self.bbox_coder = build_bbox_coder(bbox_coder)
        except Exception:
            self.bbox_coder = NMSFreeCoder(**bbox_coder)

        
        # Initialize BEV queries and object queries
        self.bev_embedding = nn.Embedding(bev_h * bev_w, in_channels)
        self.query_embedding = nn.Embedding(num_query, in_channels * 2)
        
        # Store decoder num_levels for reference points (will be extracted in _init_layers)
        self.decoder_num_levels = 4  # Default, will be updated
        
        # Real world dimensions from bbox_coder
        if bbox_coder is not None:
            if isinstance(bbox_coder, dict):
                pc_range = bbox_coder.get('pc_range', [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0])
            else:
                pc_range = getattr(bbox_coder, 'pc_range', [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0])
            self.pc_range = pc_range
            self.real_w = pc_range[3] - pc_range[0]
            self.real_h = pc_range[4] - pc_range[1]
        else:
            self.pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
            self.real_w = 102.4
            self.real_h = 102.4
        
        # Initialize classification and regression branches
        self._init_layers()
    
    def _init_layers(self):
        """Initialize classification and regression branches."""
        from mmcv.cnn import Linear
        from mmcv.cnn import bias_init_with_prob
        
        # Classification branch
        cls_branch = []
        num_reg_fcs = 2  # Default number of FC layers
        for _ in range(num_reg_fcs):
            cls_branch.append(Linear(self.in_channels, self.in_channels))
            cls_branch.append(nn.LayerNorm(self.in_channels))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(Linear(self.in_channels, self.num_classes))
        fc_cls = nn.Sequential(*cls_branch)
        
        # Regression branch
        reg_branch = []
        for _ in range(num_reg_fcs):
            reg_branch.append(Linear(self.in_channels, self.in_channels))
            reg_branch.append(nn.ReLU())
        code_size = getattr(self, 'code_size', 10)
        reg_branch.append(Linear(self.in_channels, code_size))
        reg_branch = nn.Sequential(*reg_branch)
        
        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
        
        # Number of predictions (decoder layers + 1 if as_two_stage)
        if hasattr(self.transformer, 'decoder') and self.transformer.decoder is not None:
            if hasattr(self.transformer.decoder, 'num_layers'):
                num_pred = (self.transformer.decoder.num_layers + 1) if self.as_two_stage else self.transformer.decoder.num_layers
            elif hasattr(self.transformer.decoder, 'layers'):
                num_layers = len(self.transformer.decoder.layers)
                num_pred = (num_layers + 1) if self.as_two_stage else num_layers
            else:
                num_pred = 6
        else:
            num_pred = 6  # Default
        
        if self.with_box_refine:
            self.cls_branches = _get_clones(fc_cls, num_pred)
            self.reg_branches = _get_clones(reg_branch, num_pred)
        else:
            self.cls_branches = nn.ModuleList([fc_cls for _ in range(num_pred)])
            self.reg_branches = nn.ModuleList([reg_branch for _ in range(num_pred)])
        
        # Initialize bias for classification
        if self.loss_cls_cfg.get('use_sigmoid', True):
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)
    
    def forward(self, mlvl_feats, img_metas, prev_bev=None, only_bev=False):
        """Forward function.
        
        Args:
            mlvl_feats: List of [B, num_cam, C_l, H_l, W_l] multi-view multi-level features
            img_metas: List of image meta information
            prev_bev: [B, N, C] or [N, B, C] previous BEV features
            only_bev: If True, only compute BEV features (for history frames)
            
        Returns:
            If only_bev=True: [B, N, C] BEV features
            If only_bev=False: dict with 'all_cls_scores', 'all_bbox_preds', 'bev_embed'
        """
        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype
        
        # Initialize BEV queries and object queries
        object_query_embeds = self.query_embedding.weight.to(dtype)
        bev_queries = self.bev_embedding.weight.to(dtype)
        
        # Generate positional encoding for BEV
        bev_mask = torch.zeros((bs, self.bev_h, self.bev_w), device=bev_queries.device).to(dtype)
        bev_pos = self.positional_encoding(self.bev_h, self.bev_w, bev_queries.device).to(dtype)
        bev_pos = bev_pos.repeat(bs, 1, 1)  # [B, bev_h*bev_w, embed_dims]
        # Reshape to [B, embed_dims, bev_h, bev_w] for transformer
        bev_pos = bev_pos.permute(0, 2, 1).reshape(bs, self.in_channels, self.bev_h, self.bev_w)
        
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
                reg_branches=self.reg_branches if self.with_box_refine else None,
                cls_branches=self.cls_branches if self.as_two_stage else None,
                img_metas=img_metas,
                prev_bev=prev_bev
            )
            
            bev_embed, hs, init_reference, inter_references = outputs
            
            # Process decoder outputs
            hs = hs.permute(0, 2, 1, 3)  # [num_dec_layers, B, num_query, embed_dims]
            outputs_classes = []
            outputs_coords = []
            outputs_coords_normalized = []
            
            for lvl in range(hs.shape[0]):
                if lvl == 0:
                    reference = init_reference
                else:
                    reference = inter_references[lvl - 1]
                
                # Apply inverse sigmoid to reference points
                from mmdet.models.utils.transformer import inverse_sigmoid
                reference = inverse_sigmoid(reference)
                
                # Classification and regression
                outputs_class = self.cls_branches[lvl](hs[lvl])  # [B, num_query, num_classes]
                tmp = self.reg_branches[lvl](hs[lvl])  # [B, num_query, code_size]
                normalized_tmp = tmp.clone()
                
                # Update reference points
                assert reference.shape[-1] == 3
                tmp[..., 0:2] += reference[..., 0:2]
                tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
                tmp[..., 4:5] += reference[..., 2:3]
                tmp[..., 4:5] = tmp[..., 4:5].sigmoid()
                
                normalized_tmp[..., 0:2] += reference[..., 0:2]
                normalized_tmp[..., 0:2] = normalized_tmp[..., 0:2].sigmoid()
                normalized_tmp[..., 4:5] += reference[..., 2:3]
                normalized_tmp[..., 4:5] = normalized_tmp[..., 4:5].sigmoid()

                # Convert to real world coordinates
                if hasattr(self, 'pc_range'):
                    tmp[..., 0:1] = (tmp[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0])
                    tmp[..., 1:2] = (tmp[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1])
                    tmp[..., 4:5] = (tmp[..., 4:5] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2])
                
                outputs_classes.append(outputs_class)
                outputs_coords.append(tmp)
                outputs_coords_normalized.append(normalized_tmp)
            
            outputs_classes = torch.stack(outputs_classes)  # [num_dec_layers, B, num_query, num_classes]
            outputs_coords = torch.stack(outputs_coords)  # [num_dec_layers, B, num_query, code_size]
            
            outs = {
                'bev_embed': bev_embed,
                'all_cls_scores': outputs_classes,
                'all_bbox_preds': outputs_coords,
                'enc_cls_scores': None,
                'enc_bbox_preds': None,
                'all_bbox_preds_normalized': torch.stack(outputs_coords_normalized),
            }
            
            return outs
    
    def loss(self,
             gt_bboxes_3d,
             gt_labels_3d,
             outs,
             img_metas=None,
             **kwargs) -> Dict[str, torch.Tensor]:
        """Compute losses aggregated across all decoder layers."""
        all_cls_scores = outs['all_cls_scores']
        all_bbox_preds_norm = outs.get('all_bbox_preds_normalized', outs['all_bbox_preds'])

        if gt_bboxes_3d is None or gt_labels_3d is None:
            dummy = all_cls_scores[-1].sum() * 0.0
            return {'loss_cls': dummy, 'loss_bbox': dummy}

        num_layers = all_cls_scores.shape[0]
        loss_dict: Dict[str, torch.Tensor] = {}

        for layer in range(num_layers):
            cls_scores = all_cls_scores[layer]
            bbox_preds_norm = all_bbox_preds_norm[layer]
            loss_cls_layer, loss_bbox_layer = self._loss_per_layer(
                cls_scores, bbox_preds_norm, gt_bboxes_3d, gt_labels_3d
            )
            if layer == num_layers - 1:
                loss_dict['loss_cls'] = loss_cls_layer
                loss_dict['loss_bbox'] = loss_bbox_layer
            else:
                loss_dict[f'd{layer}.loss_cls'] = loss_cls_layer
                loss_dict[f'd{layer}.loss_bbox'] = loss_bbox_layer

        return loss_dict

    def _loss_per_layer(self,
                        cls_scores: torch.Tensor,
                        bbox_preds_norm: torch.Tensor,
                        gt_bboxes_3d,
                        gt_labels_3d) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute classification and regression loss for a single decoder layer."""
        B, Q, C = cls_scores.shape
        loss_cls_total = 0.0
        loss_bbox_total = 0.0
        matched_total = 0

        alpha = self.loss_cls_cfg.get('alpha', 0.25)
        gamma = self.loss_cls_cfg.get('gamma', 2.0)
        loss_cls_weight = self.loss_cls_cfg.get('loss_weight', 2.0)
        loss_bbox_weight = self.loss_bbox_cfg.get('loss_weight', 0.25)

        for b in range(B):
            cls_b = cls_scores[b]
            bbox_b = bbox_preds_norm[b]
            gtb = gt_bboxes_3d[b]
            gtl = gt_labels_3d[b]

            if not isinstance(gtb, torch.Tensor):
                gtb = torch.tensor(gtb, device=cls_b.device, dtype=bbox_b.dtype)
            else:
                gtb = gtb.to(cls_b.device).to(bbox_b.dtype)
            if not isinstance(gtl, torch.Tensor):
                gtl = torch.tensor(gtl, device=cls_b.device, dtype=torch.long)
            else:
                gtl = gtl.to(cls_b.device).long()

            num_gt = gtb.shape[0]
            targets = torch.zeros(Q, C, device=cls_b.device, dtype=cls_b.dtype)

            if num_gt == 0:
                loss_cls_total += sigmoid_focal_loss(
                    cls_b, targets, alpha=alpha, gamma=gamma, reduction='mean'
                )
                continue

            assign_result = self.assigner.assign(bbox_b, cls_b, gtb, gtl)
            pos_mask = assign_result.gt_inds > 0

            if pos_mask.any():
                pos_inds = pos_mask.nonzero(as_tuple=False).squeeze(1)
                gt_inds = assign_result.gt_inds[pos_inds] - 1
                targets[pos_inds, gtl[gt_inds]] = 1.0

                gt_norm = normalize_bbox(gtb, getattr(self, 'pc_range', None))
                matched_total += pos_inds.numel()
                loss_bbox_total += F.l1_loss(
                    bbox_b[pos_inds, :gt_norm.size(-1)],
                    gt_norm[gt_inds],
                    reduction='sum'
                )
            loss_cls_total += sigmoid_focal_loss(
                cls_b, targets, alpha=alpha, gamma=gamma, reduction='mean'
            )

        loss_cls = (loss_cls_total / max(B, 1)) * loss_cls_weight
        if matched_total > 0:
            loss_bbox = (loss_bbox_total / matched_total) * loss_bbox_weight
        else:
            loss_bbox = bbox_preds_norm.sum() * 0.0

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
            List of detection results
        """
        preds_dicts = {
            'all_cls_scores': outs['all_cls_scores'],
            'all_bbox_preds': outs.get('all_bbox_preds_normalized', outs['all_bbox_preds'])
        }

        decoded = self.bbox_coder.decode(preds_dicts)
        results = []
        for pred in decoded:
            results.append({
                "boxes_3d": pred['bboxes'],
                "scores": pred['scores'],
                "labels": pred['labels'],
            })
        return results
