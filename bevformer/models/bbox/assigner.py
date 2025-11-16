"""Hungarian assigner aligned with BEVFormer original implementation."""
from typing import Optional

import torch

try:
    from mmdet.core.bbox.builder import BBOX_ASSIGNERS
    from mmdet.core.bbox.assigners import BaseAssigner, AssignResult
    from mmdet.core.bbox.match_costs import build_match_cost
    from mmdet.core import force_fp32
except Exception:  # pragma: no cover - fallback for environments without mmdet
    try:
        from mmdet.models.builder import ASSIGNERS as BBOX_ASSIGNERS  # type: ignore
        from mmdet.core.bbox.assigners import BaseAssigner, AssignResult  # type: ignore
        from mmdet.core.bbox.match_costs import build_match_cost  # type: ignore
    except Exception:
        BBOX_ASSIGNERS = None  # type: ignore
        build_match_cost = None  # type: ignore
        BaseAssigner = object  # type: ignore
        AssignResult = object  # type: ignore

    def force_fp32(*args, **kwargs):  # type: ignore
        def decorator(func):
            return func

        return decorator

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:  # pragma: no cover - optional dependency
    linear_sum_assignment = None


def normalize_bbox(bboxes: torch.Tensor,
                   pc_range: Optional[list] = None) -> torch.Tensor:
    """Convert bounding boxes to BEVFormer normalization format.

    Args:
        bboxes (Tensor): [..., code_size] boxes in real-world coordinates.
        pc_range (list, optional): Point cloud range, kept for API compatibility.

    Returns:
        Tensor: Normalized boxes with layout
            [cx, cy, cz, log(w), log(l), log(h), sin(rot), cos(rot), vx, vy].
    """
    cx = bboxes[..., 0:1]
    cy = bboxes[..., 1:2]
    cz = bboxes[..., 2:3]
    w = bboxes[..., 3:4].clamp(min=1e-6).log()
    l = bboxes[..., 4:5].clamp(min=1e-6).log()
    h = bboxes[..., 5:6].clamp(min=1e-6).log()
    rot = bboxes[..., 6:7]

    components = [cx, cy, cz, w, l, h, rot.sin(), rot.cos()]
    if bboxes.size(-1) > 7:
        vx = bboxes[..., 7:8]
        vy = bboxes[..., 8:9]
        components.extend([vx, vy])
    return torch.cat(components, dim=-1)



@BBOX_ASSIGNERS.register_module()
class HungarianAssigner3D(BaseAssigner):
    """Hungarian matcher used in BEVFormer DETR-style heads."""

    def __init__(self,
                    cls_cost: Optional[dict] = None,
                    reg_cost: Optional[dict] = None,
                    pc_range: Optional[list] = None):
        self.cls_cost = build_match_cost(cls_cost)
        self.reg_cost = build_match_cost(reg_cost)
        self.pc_range = pc_range

    @force_fp32(apply_to=('bbox_pred', 'cls_pred'))
    def assign(self,
                bbox_pred: torch.Tensor,
                cls_pred: torch.Tensor,
                gt_bboxes: torch.Tensor,
                gt_labels: torch.Tensor,
                gt_bboxes_ignore: Optional[torch.Tensor] = None,
                eps: float = 1e-7) -> AssignResult:
        """Assign predictions to ground truth using Hungarian algorithm.
        
        This function computes a cost matrix between predictions and ground truth,
        then uses Hungarian algorithm to find the optimal assignment that minimizes
        total cost. The cost includes classification cost, regression cost, and IoU cost.
        
        Args:
            bbox_pred: Normalized bbox predictions
                - Shape: [num_query, code_size]
                - Format: [cx, cy, cz, log(w), log(l), log(h), sin(rot), cos(rot), vx, vy]
                - Coordinates are normalized to [0, 1] range
            cls_pred: Classification predictions
                - Shape: [num_query, num_classes]
                - Logits before sigmoid
            gt_bboxes: Ground truth 3D boxes
                - Shape: [num_gt, code_size]
                - Format: [cx, cy, cz, w, l, h, rot, vx, vy] in real-world coordinates
            gt_labels: Ground truth labels
                - Shape: [num_gt]
                - Class indices (0 to num_classes-1)
            gt_bboxes_ignore: Ignored boxes (not supported)
                - Must be None
        
        Returns:
            AssignResult: Assignment result containing:
                - num_gts: Number of ground truth boxes
                - assigned_gt_inds: [num_query] - assigned GT index (0=background, 1+=GT index)
                - assigned_labels: [num_query] - assigned class labels
        """
        if linear_sum_assignment is None:
            raise ImportError('Please install scipy to use HungarianAssigner3D.')

        num_gts, num_queries = gt_bboxes.size(0), bbox_pred.size(0)
        assigned_gt_inds = bbox_pred.new_zeros(num_queries, dtype=torch.long)
        assigned_labels = bbox_pred.new_full((num_queries,), -1, dtype=torch.long)

        # Handle empty cases
        if num_gts == 0 or num_queries == 0:
            return AssignResult(num_gts, assigned_gt_inds, None, labels=assigned_labels)

        # Compute cost matrix
        normalized_gt = normalize_bbox(gt_bboxes, self.pc_range)
        cost = (self.cls_cost(cls_pred, gt_labels) + 
                self.reg_cost(bbox_pred[..., :normalized_gt.size(-1)], normalized_gt))
        
        # Hungarian matching
        row_ind, col_ind = linear_sum_assignment(cost.detach().cpu().numpy())
        row_ind = torch.from_numpy(row_ind).to(bbox_pred.device)
        col_ind = torch.from_numpy(col_ind).to(bbox_pred.device)
        
        # Assign matches (1-indexed: 0=background, 1+=GT index)
        assigned_gt_inds[row_ind] = col_ind + 1
        assigned_labels[row_ind] = gt_labels[col_ind]

        return AssignResult(num_gts, assigned_gt_inds, None, labels=assigned_labels)

