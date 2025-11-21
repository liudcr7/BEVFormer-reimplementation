# BEVFormer Model Module Files Specification

This document specifies the content and responsibilities of each file in the `bevformer/models/` directory, based on the original BEVFormer-master implementation and the project structure summary.

## Directory Structure

```
bevformer/models/
├── __init__.py                          # Module exports
├── bevformer.py                         # Main detector class
├── bevformer_head.py                    # BEVFormer detection head
├── bbox/                                 # Bounding box utilities
│   ├── __init__.py
│   ├── assigner.py                      # Hungarian matcher for assignment
│   └── coder.py                         # BBox encoding/decoding
├── transformer/                          # Transformer modules
│   ├── __init__.py
│   ├── perception_transformer.py        # Main transformer coordinator
│   ├── encoder/                          # BEVFormer encoder modules
│   │   ├── __init__.py
│   │   ├── encoders.py                  # BEVFormer encoder
│   │   ├── bevformer_layer.py           # BEVFormer encoder layer
│   │   ├── temporal_self_attention.py   # Temporal self-attention (TSA)
│   │   ├── spatial_cross_attention.py   # Spatial cross-attention (SCA)
│   │   └── ms_deform_attn_3d.py         # Multi-scale deformable attention
│   └── decoder/                          # Detection decoder modules
│       ├── __init__.py
│       ├── detr_decoder.py              # DETR-style decoder
│       └── custom_base_transformer_layer.py # Custom base transformer layer
└── utils/                                # Utility modules
    ├── __init__.py
    └── grid_mask.py                      # GridMask augmentation
```

---

## File Specifications

### 1. `bevformer/models/__init__.py`

**Purpose**: Module initialization and exports

**Content**:
- Import and export all public classes and functions
- Register modules with MMDetection/MMDetection3D registries
- Export key components:
  - `BEVFormer` detector
  - `BEVFormerHead`
  - `PerceptionTransformer`
  - `BEVFormerEncoder`, `BEVFormerLayer`
  - `TemporalSelfAttention`, `SpatialCrossAttention`
  - `DetectionTransformerDecoder`
  - `HungarianAssigner3D`, `NMSFreeCoder`
  - Loss functions and utilities

**Key Exports**:
```python
from .bevformer import BEVFormer
from .bevformer_head import BEVFormerHead
from .transformer.perception_transformer import PerceptionTransformer
from .transformer.encoder.encoders import BEVFormerEncoder
from .transformer.encoder.bevformer_layer import BEVFormerLayer
from .transformer.encoder.temporal_self_attention import TemporalSelfAttention
from .transformer.encoder.spatial_cross_attention import SpatialCrossAttention
from .transformer.encoder.ms_deform_attn_3d import MSDeformableAttention3D
from .transformer.decoder.detr_decoder import DetectionTransformerDecoder
from .transformer.decoder.custom_base_transformer_layer import CustomMSDeformableAttention
from .bbox.assigner import HungarianAssigner3D
from .bbox.coder import NMSFreeCoder
from .utils.grid_mask import GridMask
```

**Note**: `LearnedPositionalEncoding` is imported from `mmdet.models.utils.positional_encoding` for backward compatibility.

---

### 2. `bevformer/models/bevformer.py`

**Purpose**: Main detector class that orchestrates the entire BEVFormer pipeline

**Inheritance**: `MVXTwoStageDetector` (from MMDetection3D)

**Key Responsibilities**:
1. **Image Feature Extraction**
   - Extract multi-view multi-level features using backbone + neck
   - Handle temporal dimension (queue_length frames)
   - Apply GridMask augmentation during training

2. **History BEV Processing**
   - Iteratively process historical frames to build BEV features
   - Each frame uses the fused BEV from the previous frame
   - Disable gradients to save GPU memory

3. **Training Forward Pass**
   - Separate historical frames (`prev_img`) and current frame (`img`)
   - Obtain history BEV from previous frames
   - Extract current frame features
   - Forward through head and compute losses

4. **Testing Forward Pass**
   - Handle scene switching (reset `prev_bev` for new scenes)
   - Support temporal test mode (use previous BEV) or non-temporal mode
   - Handle ego motion compensation via `can_bus`
   - Update `prev_frame_info` for next frame

**Key Methods**:
- `extract_feat(img, img_metas, len_queue=None)`: Extract multi-view features
- `obtain_history_bev(imgs_queue, img_metas_list)`: Iteratively build history BEV
- `forward_train(...)`: Training forward pass
- `forward_test(...)`: Testing forward pass
- `forward(return_loss=True, **kwargs)`: Main entry point

**Key Attributes**:
- `use_grid_mask`: Whether to apply GridMask augmentation
- `enable_temporal_test`: Whether to use temporal information during inference
- `prev_frame_info`: Dictionary storing previous BEV, scene_token, position, angle
- `grid_mask`: GridMask augmentation module

**Reference**: Original implementation in `BEVFormer-master/projects/mmdet3d_plugin/bevformer/detectors/bevformer.py`

---

### 3. `bevformer/models/bevformer_head.py`

**Purpose**: Detection head that coordinates encoder, temporal/spatial attention, and decoder

**Inheritance**: Inherits from `DETRHead` (MMDetection) or `nn.Module`

**Key Responsibilities**:
1. **Query Initialization**
   - Initialize BEV queries (`bev_embedding`) for encoder
   - Initialize object queries (`query_embedding`) for decoder
   - Generate positional encodings for BEV grid

2. **Forward Pass Coordination**
   - When `only_bev=True`: Only compute BEV features (for history frames)
   - When `only_bev=False`: Full detection pipeline (encoder + decoder)
   - Process decoder outputs to generate classification and regression predictions

3. **Loss Computation**
   - Compute classification loss (Focal Loss)
   - Compute regression loss (L1 Loss)
   - Aggregate losses across all decoder layers
   - Handle auxiliary losses from intermediate decoder layers

4. **BBox Decoding**
   - Decode normalized predictions to real-world coordinates
   - Apply NMS and filtering
   - Format results for evaluation

**Key Methods**:
- `forward(mlvl_feats, img_metas, prev_bev=None, only_bev=False)`: Main forward pass
- `loss(gt_bboxes_3d, gt_labels_3d, outs, img_metas)`: Compute losses
- `get_bboxes(outs, img_metas, score_thr=0.3)`: Decode predictions to boxes
- `_init_layers()`: Initialize classification and regression branches
- `_loss_per_layer(...)`: Compute loss for a single decoder layer

**Key Attributes**:
- `bev_h`, `bev_w`: BEV grid spatial dimensions
- `num_query`: Number of object queries
- `num_classes`: Number of detection classes
- `transformer`: `PerceptionTransformer` instance
- `cls_branches`, `reg_branches`: Classification and regression heads
- `positional_encoding`: Positional encoding module
- `assigner`: Hungarian matcher for label assignment
- `bbox_coder`: BBox encoder/decoder

**Reference**: Original implementation in `BEVFormer-master/projects/mmdet3d_plugin/bevformer/dense_heads/bevformer_head.py`

---

### 4. `bevformer/models/transformer/perception_transformer.py`

**Purpose**: Main transformer coordinator that manages encoder and decoder

**Inheritance**: `BaseModule` (MMCV) or `nn.Module`

**Key Responsibilities**:
1. **BEV Feature Generation** (`get_bev_features`)
   - Process CAN bus signals (ego motion information)
   - Rotate `prev_bev` according to vehicle rotation angle
   - Add camera embeddings and level embeddings to features
   - Call encoder to generate BEV features
   - Handle ego motion shift compensation

2. **Full Forward Pass** (`forward`)
   - Generate BEV features via encoder
   - Process object queries through decoder
   - Return BEV embedding, decoder outputs, and reference points

3. **Embedding Management**
   - Level embeddings for multi-scale FPN features
   - Camera embeddings for multi-view distinction
   - CAN bus MLP for ego motion encoding

**Key Methods**:
- `get_bev_features(mlvl_feats, bev_queries, bev_h, bev_w, grid_length, bev_pos, prev_bev, img_metas)`: Generate BEV features
- `forward(mlvl_feats, bev_queries, object_query_embeds, bev_h, bev_w, grid_length, bev_pos, reg_branches, cls_branches, img_metas, prev_bev)`: Full forward pass
- `init_layers()`: Initialize embeddings and MLPs
- `init_weights()`: Initialize transformer weights

**Key Attributes**:
- `encoder`: `BEVFormerEncoder` instance
- `decoder`: `DetectionTransformerDecoder` instance
- `level_embeds`: Learnable embeddings for FPN levels
- `cams_embeds`: Learnable embeddings for camera views
- `can_bus_mlp`: MLP to process CAN bus signals
- `rotate_prev_bev`: Whether to rotate previous BEV
- `use_shift`: Whether to use shift compensation
- `use_can_bus`: Whether to use CAN bus information

**Reference**: Original implementation in `BEVFormer-master/projects/mmdet3d_plugin/bevformer/modules/transformer.py`

---

### 5. `bevformer/models/transformer/encoder/encoders.py`

**Purpose**: BEVFormer encoder that stacks multiple encoder layers

**Inheritance**: `TransformerLayerSequence` (MMCV) or `nn.Module`

**Key Responsibilities**:
1. **Reference Point Generation**
   - Generate 3D reference points for spatial cross-attention (SCA)
   - Generate 2D reference points for temporal self-attention (TSA)
   - Project 3D points to camera views

2. **Point Sampling**
   - Project 3D reference points to each camera view
   - Compute visibility masks (bev_mask)
   - Handle coordinate transformations (lidar2img)

3. **Layer Processing**
   - Stack multiple `BEVFormerLayer` blocks
   - Each layer contains TSA and SCA
   - Build hybrid reference points for temporal attention (prev_bev + current_bev)

**Key Methods**:
- `get_reference_points(H, W, Z, num_points_in_pillar, dim, bs, device, dtype)`: Generate 3D or 2D reference points
- `point_sampling(reference_points, pc_range, img_metas)`: Project 3D points to camera views
- `forward(bev_query, key, value, prev_bev, ...)`: Forward through encoder layers

**Key Attributes**:
- `layers`: List of `BEVFormerLayer` instances
- `pc_range`: Point cloud range [x_min, y_min, z_min, x_max, y_max, z_max]
- `num_points_in_pillar`: Number of points sampled along height dimension
- `return_intermediate`: Whether to return intermediate layer outputs

**Reference**: Original implementation in `BEVFormer-master/projects/mmdet3d_plugin/bevformer/modules/encoder.py`

---

### 5a. `bevformer/models/transformer/encoder/bevformer_layer.py`

**Purpose**: Single BEVFormer encoder layer containing TSA and SCA

**Inheritance**: `BaseTransformerLayer` (MMCV) or custom base class

**Key Responsibilities**:
1. **Layer Composition**
   - Contains temporal self-attention (TSA)
   - Contains spatial cross-attention (SCA)
   - Contains feedforward network (FFN) and layer normalization

2. **Forward Pass**
   - Apply TSA to fuse temporal information
   - Apply SCA to aggregate multi-view image features
   - Apply FFN for feature transformation

**Key Methods**:
- `forward(bev_query, key, value, *args, bev_pos=None, query_pos=None, ref_2d=None, ref_3d=None, bev_h=None, bev_w=None, reference_points_cam=None, mask=None, prev_bev=None, ...)`: Forward through single layer

**Key Attributes**:
- `attentions`: List containing TSA and SCA modules
- `ffns`: Feedforward network
- `norms`: Layer normalization modules

**Reference**: Original implementation in `BEVFormer-master/projects/mmdet3d_plugin/bevformer/modules/encoder.py`

---

### 6. `bevformer/models/transformer/encoder/temporal_self_attention.py`

**Purpose**: Temporal self-attention that fuses current BEV with historical BEV

**Inheritance**: `nn.Module` or attention base class

**Key Responsibilities**:
1. **Temporal Fusion**
   - Stack previous BEV and current BEV: `[prev_bev, current_bev]`
   - Use multi-scale deformable attention (MSDeformableAttention)
   - Support vehicle motion compensation via shift parameters

2. **Attention Mechanism**
   - Query: Current BEV queries
   - Key/Value: Stacked BEV features (prev + current)
   - Reference points: 2D BEV plane coordinates

**Key Methods**:
- `forward(query, key, value, prev_bev, ref_2d, ...)`: Temporal self-attention forward pass

**Key Attributes**:
- `deform_attn`: Multi-scale deformable attention module
- `num_bev_queue`: Number of BEV frames (typically 2: prev + current)
- `num_levels`: Number of levels (typically 2 for temporal attention)
- `num_points`: Number of sampling points per level

**Reference**: Original implementation in `BEVFormer-master/projects/mmdet3d_plugin/bevformer/modules/temporal_self_attention.py`

---

### 7. `bevformer/models/transformer/encoder/spatial_cross_attention.py`

**Purpose**: Spatial cross-attention that aggregates information from multi-camera images to BEV

**Inheritance**: `nn.Module` or attention base class

**Key Responsibilities**:
1. **Multi-View Aggregation**
   - Project BEV queries' 3D reference points to each camera view
   - Sample image features using deformable attention
   - Aggregate features from all camera views (mean or sum)

2. **Attention Mechanism**
   - Query: BEV queries
   - Key/Value: Multi-view image features
   - Reference points: 3D points projected to camera views

**Key Methods**:
- `forward(query, key, value, reference_points_cam, bev_mask, ...)`: Spatial cross-attention forward pass

**Key Attributes**:
- `deformable_attention`: Multi-scale deformable attention module
- `num_cams`: Number of camera views
- `num_levels`: Number of FPN levels (typically 4)
- `num_points`: Number of sampling points per level
- `pc_range`: Point cloud range for coordinate transformation

**Reference**: Original implementation in `BEVFormer-master/projects/mmdet3d_plugin/bevformer/modules/spatial_cross_attention.py`

---

### 8. `bevformer/models/transformer/encoder/ms_deform_attn_3d.py`

**Purpose**: Multi-scale deformable attention implementation for 3D BEV space

**Inheritance**: `nn.Module`

**Key Responsibilities**:
1. **Deformable Attention**
   - Implement multi-scale deformable attention mechanism
   - Support 3D reference points
   - Handle multiple feature levels (FPN levels or temporal levels)

2. **Efficient Sampling**
   - Sample features at learned offsets
   - Handle multiple scales efficiently
   - Support CUDA extensions for performance

**Key Methods**:
- `forward(query, value, reference_points, spatial_shapes, level_start_index, ...)`: Deformable attention forward pass

**Key Attributes**:
- `num_levels`: Number of feature levels
- `num_points`: Number of sampling points per level
- `num_heads`: Number of attention heads
- `embed_dims`: Embedding dimensions

**Reference**: Original implementation may use MMDetection's `MultiScaleDeformableAttention` or custom implementation

---

### 9. `bevformer/models/transformer/decoder/detr_decoder.py`

**Purpose**: DETR-style decoder for object detection

**Inheritance**: `TransformerLayerSequence` (MMCV) or `nn.Module`

**Key Responsibilities**:
1. **Object Query Processing**
   - Process object queries through multiple decoder layers
   - Each layer contains self-attention, cross-attention, and FFN
   - Generate reference points for bounding box regression

2. **Detection Output**
   - Output object queries with enhanced features
   - Generate initial and intermediate reference points
   - Support box refinement if enabled

**Key Methods**:
- `forward(query, key, value, query_pos, reference_points, ...)`: Decoder forward pass

**Key Attributes**:
- `layers`: List of decoder layers
- `num_layers`: Number of decoder layers
- `embed_dims`: Embedding dimensions

**Reference**: Similar to DETR decoder, adapted for 3D detection

---

### 9a. `bevformer/models/transformer/decoder/custom_base_transformer_layer.py`

**Purpose**: Custom base transformer layer for decoder that supports flexible attention configuration

**Inheritance**: `BaseTransformerLayer` (MMCV) or custom base class

**Key Responsibilities**:
1. **Flexible Operation Order**
   - Support custom operation order: `('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')`
   - Allow different attention configurations for self-attention and cross-attention

2. **Layer Configuration**
   - Configure self-attention for object queries
   - Configure cross-attention between queries and BEV features
   - Support feedforward network (FFN) and layer normalization

**Key Methods**:
- `forward(query, key, value, ...)`: Forward pass with flexible operation order

**Key Attributes**:
- `operation_order`: Tuple specifying the order of operations
- `attn_cfgs`: List of attention configurations
- `class_name`: `CustomMSDeformableAttention` for registration

**Reference**: Original implementation in `BEVFormer-master/projects/mmdet3d_plugin/bevformer/modules/custom_base_transformer_layer.py`

---

### 10. `bevformer/models/bbox/assigner.py`

**Purpose**: Hungarian matcher for assigning predictions to ground truth boxes

**Inheritance**: `BaseAssigner` (MMDetection) or custom base class

**Key Responsibilities**:
1. **Label Assignment**
   - Compute cost matrix between predictions and ground truth
   - Use Hungarian algorithm to find optimal assignment
   - Handle classification cost and regression cost

2. **Cost Computation**
   - Classification cost: Focal loss or cross-entropy
   - Regression cost: L1 or IoU distance
   - Handle 3D bounding boxes (center, size, rotation)

**Key Methods**:
- `assign(bbox_pred, cls_score, gt_bboxes, gt_labels, ...)`: Assign predictions to ground truth
- `compute_cost(...)`: Compute assignment cost matrix

**Key Attributes**:
- `cls_cost`: Classification cost weight
- `reg_cost`: Regression cost weight
- `iou_cost`: IoU cost weight (if used)

**Reference**: Similar to DETR's Hungarian matcher, adapted for 3D boxes

---

### 11. `bevformer/models/bbox/coder.py`

**Purpose**: Bounding box encoder/decoder for 3D detection

**Inheritance**: `BaseBBoxCoder` (MMDetection) or `nn.Module`

**Key Responsibilities**:
1. **Encoding**
   - Encode ground truth boxes to normalized format
   - Handle 3D box parameters: center (x, y, z), size (w, l, h), rotation (yaw)
   - Normalize coordinates to [0, 1] range based on `pc_range`

2. **Decoding**
   - Decode normalized predictions to real-world coordinates
   - Convert to 3D bounding box format
   - Apply NMS and score filtering

**Key Methods**:
- `encode(gt_bboxes, ...)`: Encode ground truth boxes
- `decode(preds_dicts, ...)`: Decode predictions to boxes

**Key Attributes**:
- `pc_range`: Point cloud range for normalization
- `code_size`: Size of encoded box representation (typically 10: x, y, z, w, l, h, yaw, vx, vy)

**Reference**: Original implementation in `BEVFormer-master/projects/mmdet3d_plugin/core/bbox/util.py` (normalize_bbox function)

---

### 12. `bevformer/models/utils/grid_mask.py`

**Note**: Focal loss is implemented in MMDetection (`mmdet.models.losses.FocalLoss`) and is used directly from there, not as a separate utility file.

**Purpose**: GridMask data augmentation for images

**Inheritance**: `nn.Module`

**Key Responsibilities**:
1. **Grid Masking**
   - Apply grid-shaped masks to images during training
   - Randomly mask out rectangular regions
   - Support rotation and offset

**Key Methods**:
- `forward(img)`: Apply grid mask to input images

**Key Attributes**:
- `rotate`: Rotation parameter
- `offset`: Offset parameter
- `ratio`: Mask ratio
- `mode`: Masking mode
- `prob`: Probability of applying mask

**Reference**: GridMask augmentation paper and implementation

---

## Data Flow Summary

### Training Flow:
1. **Input**: `img [B, T, V, C, H, W]` (batch, temporal, views, channels, height, width)
2. **Separate frames**: `prev_img = img[:, :-1]`, `img = img[:, -1]`
3. **History BEV**: Iteratively process `prev_img` to get `prev_bev`
4. **Feature extraction**: Extract multi-view features from current frame
5. **Encoder**: Generate BEV features via TSA + SCA
6. **Decoder**: Process object queries to get detections
7. **Loss**: Compute classification and regression losses

### Encoder Flow:
1. **Reference points**: Generate 3D (for SCA) and 2D (for TSA) reference points
2. **Projection**: Project 3D points to camera views
3. **Temporal attention**: Fuse `prev_bev` and `current_bev`
4. **Spatial attention**: Aggregate multi-view image features
5. **Output**: BEV embedding `[B, N, C]`

---

## Key Design Principles

1. **Iterative History Processing**: Each frame uses the fused BEV from the previous frame, maintaining temporal continuity.

2. **Ego Motion Compensation**: Rotate `prev_bev` and shift reference points based on CAN bus information.

3. **Multi-Scale Fusion**: Use FPN features with level embeddings and camera embeddings.

4. **Deformable Attention**: Efficiently handle large-scale feature maps with learned sampling offsets.

5. **Modular Design**: Each component (encoder, decoder, attention) is modular and can be configured independently.

---

## Implementation Notes

- Ensure compatibility with MMCV/MMDetection/MMDetection3D framework
- Register modules with appropriate registries (`DETECTORS`, `HEADS`, `TRANSFORMER`, etc.)

