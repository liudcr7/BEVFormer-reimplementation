from typing import Dict, List, Optional
import copy
import torch
from mmdet.models import DETECTORS
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from .utils.grid_mask import GridMask

@DETECTORS.register_module()
class BEVFormer(MVXTwoStageDetector):
    """BEVFormer: Learning Bird's-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers"""
    
    def __init__(self,
                 use_grid_mask: bool = True,
                 enable_temporal_test: bool = True,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        import logging
        logger = logging.getLogger(__name__)
        logger.info('BEVFormer.__init__: Starting initialization...')
        import sys
        sys.stdout.flush()
        
        logger.info('BEVFormer.__init__: Calling super().__init__()...')
        sys.stdout.flush()
        super().__init__(pts_voxel_layer, pts_voxel_encoder, pts_middle_encoder,
                         pts_fusion_layer, img_backbone, pts_backbone, img_neck,
                         pts_neck, pts_bbox_head, img_roi_head, img_rpn_head,
                         train_cfg, test_cfg, pretrained)
        logger.info('BEVFormer.__init__: super().__init__() completed')
        sys.stdout.flush()
        
        if init_cfg is not None:
            self.init_cfg = init_cfg

        self.use_grid_mask = use_grid_mask
        self.enable_temporal_test = enable_temporal_test

        logger.info('BEVFormer.__init__: Creating GridMask...')
        sys.stdout.flush()
        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5,
                                  mode=1, prob=0.7) if self.use_grid_mask else None
        logger.info('BEVFormer.__init__: Initialization completed')
        sys.stdout.flush()

        self.prev_frame_info = {
            'prev_bev': None,
            'scene_token': None,
            'prev_pos': 0,
            'prev_angle': 0,
        }

    def extract_feat(self, img: torch.Tensor, img_metas: List[dict] = None, len_queue: Optional[int] = None) -> List[torch.Tensor]:
        """Extract multi-view multi-level features from images.
        
        Args:
            img: [B, V, C, H, W] or [B*len_queue, V, C, H, W] multi-view images
            img_metas: List of image meta information
            len_queue: If provided, indicates that img contains multiple frames
            
        Returns:
            If len_queue is None: List of [B, V, C_l, H_l, W_l] feature maps
            If len_queue is provided: List of [B, len_queue, V, C_l, H_l, W_l] feature maps
        """

        B, V, C, H, W = img.size()
        x = img.view(B * V, C, H, W)

        
        # Apply GridMask if enabled and in training mode
        if self.use_grid_mask and self.grid_mask is not None:
            x = self.grid_mask(x)
        
        # Extract features
        if self.img_backbone is not None:
            feats = self.img_backbone(x)
            # if isinstance(feats, dict):
            #     feats = list(feats.values())
            # elif isinstance(feats, torch.Tensor):
            #     feats = [feats]
        else:
            feats = [x]
        
        # Apply neck
        if self.img_neck is not None:
            feats = self.img_neck(feats)
        
        # Reshape back to multi-view format
        feats_mv = []
        for f in feats:
            BN, C2, H2, W2 = f.size()
            if len_queue is not None:
                B2 = BN // (len_queue * V)
                f = f.view(B2, len_queue, V, C2, H2, W2)
            else:
                B2 = BN // V
                f = f.view(B2, V, C2, H2, W2)
            feats_mv.append(f)
        
        return feats_mv

    def obtain_history_bev(self, imgs_queue: torch.Tensor, img_metas_list: List[dict]) -> Optional[torch.Tensor]:
        """Obtain history BEV features iteratively. To save GPU memory, gradients are not calculated.
        
        This function processes historical frames sequentially, where each frame uses
        the fused BEV from the previous frame for temporal fusion. This matches the
        original BEVFormer implementation.
        
        Args:
            imgs_queue: [B, T, V, C, H, W] previous frames (T = queue_length - 1)
            img_metas_list: List of image meta information for previous frames
            
        Returns:
            prev_bev: [B, N, C] the last frame's fused BEV features or None
        """
        if imgs_queue is None or imgs_queue.shape[1] == 0:
            return None
        
        # Set to eval mode and disable gradients to save GPU memory
        self.eval()
        
        with torch.no_grad():
            prev_bev = None
            bs, len_queue, num_cams, C, H, W = imgs_queue.shape
            
            # Reshape to [bs*len_queue, num_cams, C, H, W] for batch processing
            imgs_queue = imgs_queue.reshape(bs*len_queue, num_cams, C, H, W)
            
            # Extract features for all frames at once
            img_feats_list = self.extract_feat(img=imgs_queue, len_queue=len_queue)
            
            # Iteratively process each historical frame
            # Each frame uses the fused BEV from the previous frame
            for i in range(len_queue):
                # Extract img_metas for frame i
                img_metas = [each[i] for each in img_metas_list]
                
                # Check if previous BEV exists for this frame
                if not img_metas[0].get('prev_bev_exists', False):
                    prev_bev = None
                
                # Extract features for frame i from the batch-extracted features
                img_feats = [each_scale[:, i] for each_scale in img_feats_list]
                
                # Get BEV features for this frame (only encoder, no decoder)
                prev_bev = self.pts_bbox_head(
                    img_feats, img_metas, prev_bev, only_bev=True)
        
        # Restore training mode
        self.train()
        
        return prev_bev

    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `return_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

    def forward_train(self, 
                     img=None, 
                     img_metas=None,
                     gt_bboxes_3d=None, 
                     gt_labels_3d=None,
                     gt_bboxes_ignore=None,
                     **kwargs) -> Dict[str, torch.Tensor]:
        """Forward function for training.
        
        Args:
            img: [B, T, V, C, H, W] multi-view images with temporal dimension
            img_metas: List of image meta information (nested list: [B][T] for each frame)
            gt_bboxes_3d: List of ground truth 3D bounding boxes
            gt_labels_3d: List of ground truth labels
            gt_bboxes_ignore: List of ignored bounding boxes
            
        Returns:
            Dict of losses
        """
        # Extract queue length
        len_queue = img.size(1)  # T dimension
        
        # Separate previous frames and current frame
        prev_img = img[:, :-1, ...]  # [B, T-1, V, C, H, W]
        img = img[:, -1, ...]  # [B, V, C, H, W]
        
        # Copy previous img_metas
        prev_img_metas = copy.deepcopy(img_metas)
        
        # Obtain history BEV from previous frames
        # prev_img_metas should be a list of lists: [B][T] for each frame
        prev_bev = self.obtain_history_bev(prev_img, prev_img_metas)
        
        # Extract current frame's img_metas
        img_metas = [each[len_queue-1] for each in img_metas]
        
        # Check if previous BEV exists
        if img_metas is not None and len(img_metas) > 0:
            if isinstance(img_metas[0], dict):
                prev_bev_exists = img_metas[0].get('prev_bev_exists', False)
            elif hasattr(img_metas[0], 'data'):
                prev_bev_exists = img_metas[0].data.get('prev_bev_exists', False)
            else:
                prev_bev_exists = False
            
            if not prev_bev_exists:
                prev_bev = None
        
        # Extract features for current frame
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        
        # Forward through head to get predictions
        outs = self.pts_bbox_head.forward(
            img_feats,
            img_metas,
            prev_bev=prev_bev,
            only_bev=False
        )

        # Compute losses using head's loss function
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs, img_metas=img_metas)
        return losses

    def forward_test(self, img=None, img_metas=None, **kwargs):
        """Forward function for testing.
        
        Args:
            img: [B, T, V, C, H, W] or [B, V, C, H, W] multi-view images
            img_metas: List of image meta information (nested list for test time augmentation)
            
        Returns:
            List of detection results
        """
        # Handle test time augmentation format (double nested)
        if img_metas is not None and len(img_metas) > 0:
            if isinstance(img_metas[0], list):
                # Test time augmentation format: list[list[dict]]
                img_metas = img_metas[0]
                if img is not None:
                    img = img[0] if isinstance(img, list) else img
        
        # Check scene token for scene switching
        if img_metas is not None and len(img_metas) > 0:
            if isinstance(img_metas[0], dict) and 'scene_token' in img_metas[0]:
                if img_metas[0]['scene_token'] != self.prev_frame_info['scene_token']:
                    # First sample of each scene, reset prev_bev
                    self.prev_frame_info['prev_bev'] = None
                self.prev_frame_info['scene_token'] = img_metas[0]['scene_token']
        
        # Do not use temporal information if temporal test is disabled
        if not self.enable_temporal_test:
            self.prev_frame_info['prev_bev'] = None
        
        # Get previous BEV from prev_frame_info
        prev_bev = self.prev_frame_info['prev_bev']
        
        # Handle can_bus for ego motion compensation
        if img_metas is not None and len(img_metas) > 0:
            if isinstance(img_metas[0], dict) and 'can_bus' in img_metas[0]:
                # Get the delta of ego position and angle between two timestamps
                tmp_pos = copy.deepcopy(img_metas[0]['can_bus'][:3])
                tmp_angle = copy.deepcopy(img_metas[0]['can_bus'][-1])
                
                if prev_bev is not None:
                    # Calculate delta relative to previous frame
                    img_metas[0]['can_bus'][:3] -= self.prev_frame_info['prev_pos']
                    img_metas[0]['can_bus'][-1] -= self.prev_frame_info['prev_angle']
                else:
                    # First frame, set to zero
                    img_metas[0]['can_bus'][:3] = 0
                    img_metas[0]['can_bus'][-1] = 0
                
                # Store current position and angle for next frame
                self.prev_frame_info['prev_pos'] = tmp_pos
                self.prev_frame_info['prev_angle'] = tmp_angle
        
        # Handle both temporal and non-temporal inputs
        if img is not None and img.dim() == 6:
            # Temporal input: [B, T, V, C, H, W]
            len_queue = img.size(1)
            prev_img = img[:, :-1, ...]
            img = img[:, -1, ...]
            
            # Get previous BEV from history frames
            prev_img_metas = [each[:-1] if isinstance(each, list) else [each] * (len_queue-1) 
                             for each in img_metas]
            prev_bev = self.obtain_history_bev(prev_img, prev_img_metas)
            
            # Current frame metas
            img_metas = [each[len_queue-1] if isinstance(each, list) else each for each in img_metas]
            
            # Check prev_bev_exists
            if img_metas is not None and len(img_metas) > 0:
                if isinstance(img_metas[0], dict):
                    prev_bev_exists = img_metas[0].get('prev_bev_exists', False)
                elif hasattr(img_metas[0], 'data'):
                    prev_bev_exists = img_metas[0].data.get('prev_bev_exists', False)
                else:
                    prev_bev_exists = False
                
                if not prev_bev_exists:
                    prev_bev = None
        
        # Extract features for current frame
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        
        # Forward through head
        outs = self.pts_bbox_head.forward(
            img_feats,
            img_metas,
            prev_bev=prev_bev,
            only_bev=False
        )

        # Extract BEV features for next frame
        new_prev_bev = outs.get('bev_embed', None)

        # Decode boxes
        results = self.pts_bbox_head.get_bboxes(outs, img_metas=img_metas)
        
        # Update prev_frame_info with new BEV features for next frame
        if new_prev_bev is not None:
            self.prev_frame_info['prev_bev'] = new_prev_bev
        
        return results
