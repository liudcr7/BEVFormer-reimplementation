from typing import Any, Dict, List
from mmdet3d.datasets import NuScenesDataset
from mmdet.datasets import DATASETS
import torch
import numpy as np

@DATASETS.register_module()
class CustomNuScenesDataset(NuScenesDataset):
    CLASSES = (
        'car', 'truck', 'bus', 'trailer', 'construction_vehicle',
        'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone', 'barrier'
    )

    def prepare_train_data(self, index: int) -> Dict[str, Any]:
        """Prepare training data with camera parameters."""
        data = super().prepare_train_data(index)
        # img_metas is already in DataContainer format
        if hasattr(data['img_metas'], 'data'):
            img_metas = data['img_metas'].data[0]
        else:
            img_metas = data['img_metas']
        
        # Extract camera parameters from img_metas
        # cam2img and lidar2cam are lists over V views
        cam2img_list = img_metas.get('cam2img', [])  # list[V] of 4x4 or 3x3
        lidar2cam_list = img_metas.get('lidar2cam', [])  # list[V] of 4x4
        
        # Store as numpy arrays in img_metas (will be converted to tensors in model)
        # This ensures they're properly handled by DataContainer
        img_metas['cam2img'] = cam2img_list
        img_metas['lidar2cam'] = lidar2cam_list
        
        # Update img_metas in data
        if hasattr(data['img_metas'], 'data'):
            data['img_metas'].data[0] = img_metas
        else:
            data['img_metas'] = img_metas
        
        return data

    def prepare_test_data(self, index: int) -> Dict[str, Any]:
        """Prepare test data with camera parameters."""
        data = super().prepare_test_data(index)
        if hasattr(data['img_metas'], 'data'):
            img_metas = data['img_metas'].data[0]
        else:
            img_metas = data['img_metas']
        
        cam2img_list = img_metas.get('cam2img', [])
        lidar2cam_list = img_metas.get('lidar2cam', [])
        
        img_metas['cam2img'] = cam2img_list
        img_metas['lidar2cam'] = lidar2cam_list
        
        if hasattr(data['img_metas'], 'data'):
            data['img_metas'].data[0] = img_metas
        else:
            data['img_metas'] = img_metas
        
        return data
