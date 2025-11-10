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


if BBOX_ASSIGNERS is not None and build_match_cost is not None:

    @BBOX_ASSIGNERS.register_module()
    class HungarianAssigner3D(BaseAssigner):
        """Hungarian matcher used in BEVFormer DETR-style heads."""

        def __init__(self,
                     cls_cost: Optional[dict] = None,
                     reg_cost: Optional[dict] = None,
                     iou_cost: Optional[dict] = None,
                     pc_range: Optional[list] = None):
            cls_cost = cls_cost or dict(type='FocalLossCost',
                                        weight=2.0,
                                        alpha=0.25,
                                        gamma=2.0)
            reg_cost = reg_cost or dict(type='BBoxL1Cost', weight=0.25)
            iou_cost = iou_cost or dict(type='IoUCost', weight=0.0)

            self.cls_cost = build_match_cost(cls_cost)
            self.reg_cost = build_match_cost(reg_cost)
            self.iou_cost = build_match_cost(iou_cost)
            self.pc_range = pc_range

        @force_fp32(apply_to=('bbox_pred', 'cls_pred'))
        def assign(self,
                   bbox_pred: torch.Tensor,
                   cls_pred: torch.Tensor,
                   gt_bboxes: torch.Tensor,
                   gt_labels: torch.Tensor,
                   gt_bboxes_ignore: Optional[torch.Tensor] = None,
                   eps: float = 1e-7) -> AssignResult:
            if linear_sum_assignment is None:
                raise ImportError('Please install scipy to use HungarianAssigner3D.')

            assert gt_bboxes_ignore is None, \
                'HungarianAssigner3D does not support gt_bboxes_ignore.'

            num_gts, num_bboxes = gt_bboxes.size(0), bbox_pred.size(0)

            assigned_gt_inds = bbox_pred.new_full((num_bboxes,), -1, dtype=torch.long)
            assigned_labels = bbox_pred.new_full((num_bboxes,), -1, dtype=torch.long)

            if num_gts == 0 or num_bboxes == 0:
                if num_gts == 0:
                    assigned_gt_inds[:] = 0
                return AssignResult(num_gts, assigned_gt_inds, None, labels=assigned_labels)

            cls_cost = self.cls_cost(cls_pred, gt_labels)
            normalized_gt = normalize_bbox(gt_bboxes, self.pc_range)
            reg_cost = self.reg_cost(bbox_pred[..., :normalized_gt.size(-1)],
                                     normalized_gt)
            iou_cost = self.iou_cost(bbox_pred[..., :normalized_gt.size(-1)],
                                     normalized_gt)

            cost = cls_cost + reg_cost + iou_cost
            cost = cost.detach().cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(cost)

            assigned_gt_inds[:] = 0
            row_ind = torch.from_numpy(row_ind).to(bbox_pred.device)
            col_ind = torch.from_numpy(col_ind).to(bbox_pred.device)
            assigned_gt_inds[row_ind] = col_ind + 1
            assigned_labels[row_ind] = gt_labels[col_ind]

            return AssignResult(num_gts, assigned_gt_inds, None, labels=assigned_labels)

else:  # pragma: no cover - executed only when mmdet is unavailable

    class HungarianAssigner3D:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError('mmdet is required to build HungarianAssigner3D.')
