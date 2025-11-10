# BEVFormer-master Original Project Structure Summary

## Project Overview

BEVFormer is a Bird's-Eye-View (BEV) representation learning framework based on multi-camera images, using spatiotemporal Transformers for 3D object detection. This project is the original official implementation, based on the MMDetection3D framework.

## Directory Structure

```
BEVFormer-master/
├── docs/                          # Documentation directory
│   ├── install.md                # Installation instructions
│   ├── getting_started.md        # Quick start guide
│   ├── prepare_dataset.md        # Dataset preparation
│   └── can_bus.ipynb            # CAN bus data explanation
├── figs/                          # Image resources
│   ├── arch.png                  # Architecture diagram
│   └── sota_results.png          # Results comparison chart
├── projects/                      # Core code directory
│   ├── configs/                  # Configuration files
│   │   ├── _base_/              # Base configurations
│   │   │   ├── datasets/        # Dataset configurations
│   │   │   ├── models/          # Model configurations
│   │   │   └── schedules/       # Training strategy configurations
│   │   ├── bevformer/           # BEVFormer configurations
│   │   ├── bevformer_fp16/      # FP16 version configurations
│   │   └── bevformerv2/         # BEVFormerV2 configurations
│   └── mmdet3d_plugin/           # MMDet3D plugin (core implementation)
│       ├── bevformer/           # BEVFormer core modules
│       ├── datasets/            # Dataset related
│       ├── core/                # Core functionality (bbox, evaluation, etc.)
│       ├── dd3d/                # DD3D related (for pretraining)
│       └── models/              # Model utilities (backbone, utils, etc.)
├── tools/                         # Utility scripts
│   ├── train.py                 # Training script
│   ├── test.py                  # Testing script
│   ├── data_converter/          # Data conversion tools
│   ├── analysis_tools/          # Analysis tools
│   └── misc/                    # Miscellaneous tools
└── README.md                      # Project description
```

## Core Module Architecture

### 1. Detector Layer (`bevformer/detectors/`)

#### `bevformer.py` - Main Detector
- **Inheritance**: `MVXTwoStageDetector` (MMDet3D)
- **Core Functions**:
  - Image feature extraction (`extract_img_feat`)
  - History BEV feature acquisition (`obtain_history_bev`)
  - Training forward propagation (`forward_train`)
  - Testing forward propagation (`forward_test`)

**Key Methods**:
```python
def obtain_history_bev(self, imgs_queue, img_metas_list):
    """Iteratively obtain history BEV features
    - Input: [B, T, V, C, H, W] history image queue
    - Process: Process frame by frame, each frame uses the fused BEV from the previous frame as prev_bev
    - Output: BEV features of the last frame
    """
    for i in range(len_queue):
        prev_bev = self.pts_bbox_head(
            img_feats, img_metas, prev_bev, only_bev=True)
    return prev_bev

def forward_train(self, img, img_metas, ...):
    """Training forward propagation
    - Separate history frames and current frame: prev_img = img[:, :-1], img = img[:, -1]
    - Obtain history BEV: prev_bev = self.obtain_history_bev(prev_img, ...)
    - Extract current frame features: img_feats = self.extract_feat(img, ...)
    - Forward propagation: losses = self.forward_pts_train(img_feats, ..., prev_bev)
    """
```

### 2. Head Layer (`bevformer/dense_heads/`)

#### `bevformer_head.py` - BEVFormer Detection Head
- **Inheritance**: `DETRHead` (MMDet)
- **Core Components**:
  - `transformer`: `PerceptionTransformer` instance
  - `bev_embedding`: BEV query embeddings
  - `query_embedding`: Object query embeddings
  - `cls_branches`: Classification branches
  - `reg_branches`: Regression branches

**Key Methods**:
```python
def forward(self, mlvl_feats, img_metas, prev_bev=None, only_bev=False):
    """Forward propagation
    - only_bev=True: Only compute BEV features (for history frame processing)
    - only_bev=False: Complete detection pipeline (encoder + decoder)
    """
    if only_bev:
        return self.transformer.get_bev_features(...)
    else:
        outputs = self.transformer(...)  # Complete transformer
```

### 3. Transformer Layer (`bevformer/modules/`)

#### `transformer.py` - PerceptionTransformer
- **Core Function**: Coordinate encoder and decoder
- **Key Methods**:
  - `get_bev_features()`: Get BEV features (encoder only)
  - `forward()`: Complete forward propagation (encoder + decoder)

**Key Implementation**:
```python
def get_bev_features(self, mlvl_feats, bev_queries, ...):
    """Get BEV features
    1. Process can_bus signals (vehicle motion information)
    2. Rotate prev_bev (according to vehicle rotation angle)
    3. Add camera embeddings and level embeddings
    4. Call encoder to get BEV features
    """
    # Rotate prev_bev
    if self.rotate_prev_bev:
        rotation_angle = img_metas[i]['can_bus'][-1]
        tmp_prev_bev = rotate(tmp_prev_bev, rotation_angle, ...)
    
    # Add can_bus signals
    can_bus = self.can_bus_mlp(can_bus)
    bev_queries = bev_queries + can_bus * self.use_can_bus
    
    # Call encoder
    bev_embed = self.encoder(bev_queries, feat_flatten, ...)
```

#### `encoder.py` - BEVFormerEncoder
- **Inheritance**: `TransformerLayerSequence`
- **Core Function**: Stack multiple BEVFormerLayer blocks
- **Key Methods**:
  - `get_reference_points()`: Generate reference points (3D for SCA, 2D for TSA)
  - `point_sampling()`: Project 3D reference points to each camera view
  - `forward()`: Forward propagation

**Key Implementation**:
```python
def forward(self, bev_query, key, value, prev_bev=None, ...):
    """Encoder forward propagation
    1. Generate 3D reference points (for spatial cross-attention)
    2. Generate 2D reference points (for temporal self-attention)
    3. Project 3D points to each camera view
    4. Build hybrid_ref_2d (containing prev_bev and current_bev)
    5. Process layer by layer (each layer contains TSA and SCA)
    """
    # Build hybrid reference points
    if prev_bev is not None:
        prev_bev = torch.stack([prev_bev, bev_query], 1)  # [bs*2, len_bev, -1]
        hybird_ref_2d = torch.stack([shift_ref_2d, ref_2d], 1)  # [bs*2, len_bev, ...]
    
    for layer in self.layers:
        output = layer(bev_query, key, value, 
                      ref_2d=hybird_ref_2d,
                      ref_3d=ref_3d,
                      prev_bev=prev_bev, ...)
```

#### `custom_base_transformer_layer.py` - Custom Transformer Layer Base Class
- **Core Function**: Support flexible attention configuration
- **operation_order**: `('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')`
- **attn_cfgs**: Contains two attention configurations
  - `[0]`: TemporalSelfAttention (self_attn)
  - `[1]`: SpatialCrossAttention (cross_attn)

#### `temporal_self_attention.py` - Temporal Self-Attention
- **Function**: Fuse current BEV with history BEV
- **Key Features**:
  - Uses `num_bev_queue=2` (prev_bev and current_bev)
  - Uses multi-scale deformable attention (MSDeformableAttention)
  - Supports vehicle motion compensation (via shift parameter)

**Key Implementation**:
```python
def forward(self, query, key, value, prev_bev=None, ref_2d=None, ...):
    """Temporal self-attention forward propagation
    - query: Current BEV queries
    - prev_bev: History BEV features (already rotated and aligned)
    - ref_2d: 2D reference points (BEV plane coordinates)
    """
    # Build value: [prev_bev, current_bev]
    if prev_bev is not None:
        value = torch.stack([prev_bev, query], dim=1)  # [bs, 2, N, C]
    else:
        value = torch.stack([query, query], dim=1)
    
    # Use deformable attention
    output = self.deform_attn(query, value, ref_2d, ...)
```

#### `spatial_cross_attention.py` - Spatial Cross-Attention
- **Function**: Aggregate information from multi-camera image features to BEV
- **Key Features**:
  - Project 3D reference points of BEV queries to each camera view
  - Use deformable attention to sample image features
  - Multi-view fusion (mean or sum)

**Key Implementation**:
```python
def forward(self, query, key, value, reference_points_cam=None, ...):
    """Spatial cross-attention forward propagation
    - query: BEV queries
    - reference_points_cam: Reference points projected to each camera [num_cam, B, N, D, 2]
    - bev_mask: Visibility mask
    """
    # Process camera by camera
    for cam_id in range(num_cams):
        # Extract reference points and features for this camera
        ref_points_cam = reference_points_cam[cam_id]
        feat_cam = value[cam_id]
        
        # Deformable attention
        output_cam = self.deformable_attention(
            query, feat_cam, ref_points_cam, ...)
        outputs.append(output_cam)
    
    # Multi-view fusion
    output = torch.stack(outputs).mean(dim=0)
```

#### `decoder.py` - Detection Decoder
- **Function**: Decode 3D detection results from BEV features
- **Structure**: Similar to DETR decoder
  - Self-attention: Interaction between queries
  - Cross-attention: Interaction between queries and BEV features
  - FFN: Feedforward network

## Data Flow

### Training Flow

```
Input: img [B, T, V, C, H, W]
  ↓
1. Separate history frames and current frame
   - prev_img = img[:, :-1]  # [B, T-1, V, C, H, W]
   - img = img[:, -1]        # [B, V, C, H, W]
  ↓
2. Obtain history BEV (obtain_history_bev)
   for t in range(T-1):
     - Extract features: img_feats = extract_feat(prev_img[:, t])
     - Get BEV: prev_bev = transformer.get_bev_features(..., prev_bev)
   # Note: Each frame uses the fused BEV from the previous frame as prev_bev
  ↓
3. Extract current frame features
   - img_feats = extract_feat(img)  # List of [B, V, C, H_l, W_l]
  ↓
4. Complete detection pipeline
   - BEV features: bev_embed = encoder(bev_queries, img_feats, prev_bev)
   - Detection results: outputs = decoder(bev_embed, object_queries)
   - Loss computation: losses = loss(outputs, gt_bboxes_3d, gt_labels_3d)
```

### Encoder Internal Flow

```
BEV Queries [B, N, C]
  ↓
1. Process prev_bev
   - Rotate alignment: prev_bev = rotate(prev_bev, can_bus[-1])
   - Build hybrid: [prev_bev, current_bev] → [B*2, N, C]
  ↓
2. Generate reference points
   - ref_3d: 3D spatial reference points [B, N, D, 3] (for SCA)
   - ref_2d: 2D BEV plane reference points [B, N, 2] (for TSA)
   - Project: ref_3d → reference_points_cam [num_cam, B, N, D, 2]
  ↓
3. Process layer by layer (BEVFormerLayer × 6)
   for layer in layers:
     a. Temporal Self-Attention
        - Input: query, prev_bev, ref_2d
        - Output: Fused BEV features
     b. Spatial Cross-Attention
        - Input: query, img_feats, reference_points_cam
        - Output: BEV features aggregated from image features
     c. FFN
        - Output: Final BEV features
  ↓
Output: BEV Embed [B, N, C]
```

## Key Design Principles

### 1. Iterative History BEV Construction
- **Principle**: Each frame's BEV fuses the fused BEV from the previous frame, rather than processing independently
- **Implementation**: Loop processing in `obtain_history_bev`, `prev_bev` continuously updates
- **Advantage**: Maintains temporal continuity and accumulates historical information

### 2. Vehicle Motion Compensation
- **Rotation**: Rotate prev_bev according to can_bus yaw angle
- **Translation**: Calculate shift based on can_bus x, y, adjust reference points
- **Implementation Location**: `PerceptionTransformer.get_bev_features()`

### 3. Multi-Scale Feature Fusion
- **FPN Features**: 4 scales of feature maps
- **Level Embeddings**: `level_embeds` distinguish different scales
- **Camera Embeddings**: `cams_embeds` distinguish different camera views

### 4. Deformable Attention
- **TSA**: 2 levels (prev_bev, current_bev), each level samples multiple points
- **SCA**: 4 levels (FPN features), each level samples multiple points
- **Advantage**: Efficiently handle large-scale feature maps

## Configuration File Structure

### Model Configuration (`configs/bevformer/bevformer_base.py`)
```python
model = dict(
    type='BEVFormer',
    img_backbone=dict(...),      # ResNet + DCN
    img_neck=dict(...),          # FPN
    pts_bbox_head=dict(
        type='BEVFormerHead',
        transformer=dict(
            type='PerceptionTransformer',
            encoder=dict(
                type='BEVFormerEncoder',
                num_layers=6,
                transformerlayers=dict(
                    type='BEVFormerLayer',
                    attn_cfgs=[
                        dict(type='TemporalSelfAttention', ...),  # TSA
                        dict(type='SpatialCrossAttention', ...),  # SCA
                    ],
                ),
            ),
            decoder=dict(...),
        ),
    ),
)
```

## Summary

BEVFormer's core innovations are:
1. **Spatiotemporal Fusion**: Fuse history BEV through TSA, aggregate multi-view image features through SCA
2. **Iterative Processing**: Iterative construction of history BEV maintains temporal continuity
3. **Motion Compensation**: Compensate vehicle motion through can_bus information
4. **Deformable Attention**: Efficiently handle large-scale feature maps

Special attention needed during reimplementation:
- Each encoder layer must contain TSA and SCA
- Iterative construction logic of history BEV
- Implementation location of vehicle motion compensation
- Generation and projection of reference points

