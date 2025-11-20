"""BBox coder for BEVFormer detection head."""
from typing import List, Optional

import torch
from mmdet.core.bbox import BaseBBoxCoder
from mmdet.core.bbox.builder import BBOX_CODERS


def denormalize_bbox(normalized_bboxes, pc_range):
    # rotation 
    rot_sine = normalized_bboxes[..., 6:7]
    rot_cosine = normalized_bboxes[..., 7:8]
    rot = torch.atan2(rot_sine, rot_cosine)

    # center in the bev - already in real-world coordinates
    # Format: [cx, cy, log(w), log(l), cz, log(h), sin(rot), cos(rot), vx, vy]
    cx = normalized_bboxes[..., 0:1]  # Already real-world
    cy = normalized_bboxes[..., 1:2]  # Already real-world
    cz = normalized_bboxes[..., 4:5]  # Already real-world, at index 4 (matching normalize_bbox format)
   
    # size - convert from log scale to real scale
    w = normalized_bboxes[..., 2:3].exp()  # log(w) -> w
    l = normalized_bboxes[..., 3:4].exp()  # log(l) -> l
    h = normalized_bboxes[..., 5:6].exp()  # log(h) -> h
    
    if normalized_bboxes.size(-1) > 8:
        # velocity 
        vx = normalized_bboxes[:, 8:9]
        vy = normalized_bboxes[:, 9:10]
        denormalized_bboxes = torch.cat([cx, cy, cz, w, l, h, rot, vx, vy], dim=-1)
    else:
        denormalized_bboxes = torch.cat([cx, cy, cz, w, l, h, rot], dim=-1)
    return denormalized_bboxes

@BBOX_CODERS.register_module()
class NMSFreeCoder(BaseBBoxCoder):
    """BBox coder that keeps top-k predictions without NMS."""

    def __init__(self,
                 pc_range: List[float],
                 post_center_range: Optional[List[float]] = None,
                 max_num: int = 100,
                 num_classes: int = 10,
                 voxel_size: Optional[List[float]] = None):
        super().__init__()
        self.pc_range = pc_range
        self.post_center_range = post_center_range
        self.max_num = max_num
        self.num_classes = num_classes
        self.voxel_size = voxel_size

    def encode(self, bboxes, gt_bboxes):
        """Encode bboxes to targets (not used in BEVFormer, but required by BaseBBoxCoder).
        
        Args:
            bboxes: Source bboxes
            gt_bboxes: Target bboxes
            
        Returns:
            Encoded bboxes (placeholder implementation)
        """
        # BEVFormer doesn't use encode, but we need to implement it for BaseBBoxCoder
        return gt_bboxes

    def decode(self, preds_dicts):
        """Decode classification and regression outputs to 3D bounding boxes.
        
        Args:
            preds_dicts: Dictionary containing predictions
                - 'all_cls_scores': [num_dec_layers, B, num_query, num_classes]
                - 'all_bbox_preds': [num_dec_layers, B, num_query, code_size]
        
        Returns:
            List[dict]: Decoded results for each sample in batch
        """
        cls_scores = preds_dicts['all_cls_scores'][-1]  # [B, num_query, num_classes]
        bbox_preds = preds_dicts['all_bbox_preds'][-1]  # [B, num_query, code_size]
        
        results = []
        for i in range(cls_scores.size(0)):
            # Get top-k predictions across all queries and classes
            scores, indices = cls_scores[i].sigmoid().view(-1).topk(self.max_num)
            labels = indices % self.num_classes
            bbox_indices = indices // self.num_classes
            
            # Denormalize selected bbox predictions
            boxes3d = denormalize_bbox(bbox_preds[i][bbox_indices], self.pc_range)
            
            # Apply center range filter
            if self.post_center_range is not None:
                center_range = torch.tensor(
                    self.post_center_range,
                    device=boxes3d.device,
                    dtype=boxes3d.dtype
                )
                mask = (boxes3d[..., :3] >= center_range[:3]).all(-1)
                mask &= (boxes3d[..., :3] <= center_range[3:]).all(-1)
                boxes3d, scores, labels = boxes3d[mask], scores[mask], labels[mask]
            
            results.append({
                'bboxes': boxes3d,
                'scores': scores,
                'labels': labels,
            })
        
        return results
