# BEVFormer-master 原版项目结构总结

## 项目概述

BEVFormer是一个基于多相机图像的鸟瞰图（BEV）表示学习框架，使用时空Transformer进行3D目标检测。本项目是原版官方实现，基于MMDetection3D框架。

## 目录结构

```
BEVFormer-master/
├── docs/                          # 文档目录
│   ├── install.md                # 安装说明
│   ├── getting_started.md        # 快速开始
│   ├── prepare_dataset.md        # 数据集准备
│   └── can_bus.ipynb            # CAN总线数据说明
├── figs/                          # 图片资源
│   ├── arch.png                  # 架构图
│   └── sota_results.png          # 结果对比图
├── projects/                      # 核心代码目录
│   ├── configs/                  # 配置文件
│   │   ├── _base_/              # 基础配置
│   │   │   ├── datasets/        # 数据集配置
│   │   │   ├── models/          # 模型配置
│   │   │   └── schedules/       # 训练策略配置
│   │   ├── bevformer/           # BEVFormer配置
│   │   ├── bevformer_fp16/      # FP16版本配置
│   │   └── bevformerv2/         # BEVFormerV2配置
│   └── mmdet3d_plugin/           # MMDet3D插件（核心实现）
│       ├── bevformer/           # BEVFormer核心模块
│       ├── datasets/            # 数据集相关
│       ├── core/                # 核心功能（bbox、evaluation等）
│       ├── dd3d/                # DD3D相关（用于预训练）
│       └── models/              # 模型工具（backbone、utils等）
├── tools/                         # 工具脚本
│   ├── train.py                 # 训练脚本
│   ├── test.py                  # 测试脚本
│   ├── data_converter/          # 数据转换工具
│   ├── analysis_tools/          # 分析工具
│   └── misc/                    # 杂项工具
└── README.md                      # 项目说明
```

## 核心模块架构

### 1. Detector层 (`bevformer/detectors/`)

#### `bevformer.py` - 主检测器
- **继承**: `MVXTwoStageDetector` (MMDet3D)
- **核心功能**:
  - 图像特征提取 (`extract_img_feat`)
  - 历史BEV特征获取 (`obtain_history_bev`)
  - 训练前向传播 (`forward_train`)
  - 测试前向传播 (`forward_test`)

**关键方法**:
```python
def obtain_history_bev(self, imgs_queue, img_metas_list):
    """迭代获取历史BEV特征
    - 输入: [B, T, V, C, H, W] 历史图像队列
    - 过程: 逐帧处理，每帧使用前一帧的融合BEV作为prev_bev
    - 输出: 最后一帧的BEV特征
    """
    for i in range(len_queue):
        prev_bev = self.pts_bbox_head(
            img_feats, img_metas, prev_bev, only_bev=True)
    return prev_bev

def forward_train(self, img, img_metas, ...):
    """训练前向传播
    - 分离历史帧和当前帧: prev_img = img[:, :-1], img = img[:, -1]
    - 获取历史BEV: prev_bev = self.obtain_history_bev(prev_img, ...)
    - 提取当前帧特征: img_feats = self.extract_feat(img, ...)
    - 前向传播: losses = self.forward_pts_train(img_feats, ..., prev_bev)
    """
```

### 2. Head层 (`bevformer/dense_heads/`)

#### `bevformer_head.py` - BEVFormer检测头
- **继承**: `DETRHead` (MMDet)
- **核心组件**:
  - `transformer`: `PerceptionTransformer`实例
  - `bev_embedding`: BEV查询嵌入
  - `query_embedding`: 目标查询嵌入
  - `cls_branches`: 分类分支
  - `reg_branches`: 回归分支

**关键方法**:
```python
def forward(self, mlvl_feats, img_metas, prev_bev=None, only_bev=False):
    """前向传播
    - only_bev=True: 仅计算BEV特征（用于历史帧处理）
    - only_bev=False: 完整检测流程（encoder + decoder）
    """
    if only_bev:
        return self.transformer.get_bev_features(...)
    else:
        outputs = self.transformer(...)  # 完整transformer
```

### 3. Transformer层 (`bevformer/modules/`)

#### `transformer.py` - PerceptionTransformer
- **核心功能**: 协调encoder和decoder
- **关键方法**:
  - `get_bev_features()`: 获取BEV特征（仅encoder）
  - `forward()`: 完整前向传播（encoder + decoder）

**关键实现**:
```python
def get_bev_features(self, mlvl_feats, bev_queries, ...):
    """获取BEV特征
    1. 处理can_bus信号（车辆运动信息）
    2. 旋转prev_bev（根据车辆旋转角度）
    3. 添加相机嵌入和层级嵌入
    4. 调用encoder获取BEV特征
    """
    # 旋转prev_bev
    if self.rotate_prev_bev:
        rotation_angle = img_metas[i]['can_bus'][-1]
        tmp_prev_bev = rotate(tmp_prev_bev, rotation_angle, ...)
    
    # 添加can_bus信号
    can_bus = self.can_bus_mlp(can_bus)
    bev_queries = bev_queries + can_bus * self.use_can_bus
    
    # 调用encoder
    bev_embed = self.encoder(bev_queries, feat_flatten, ...)
```

#### `encoder.py` - BEVFormerEncoder
- **继承**: `TransformerLayerSequence`
- **核心功能**: 堆叠多个BEVFormerLayer
- **关键方法**:
  - `get_reference_points()`: 生成参考点（3D用于SCA，2D用于TSA）
  - `point_sampling()`: 将3D参考点投影到各相机视图
  - `forward()`: 前向传播

**关键实现**:
```python
def forward(self, bev_query, key, value, prev_bev=None, ...):
    """Encoder前向传播
    1. 生成3D参考点（用于空间交叉注意力）
    2. 生成2D参考点（用于时间自注意力）
    3. 将3D点投影到各相机视图
    4. 构建hybrid_ref_2d（包含prev_bev和current_bev）
    5. 逐层处理（每层包含TSA和SCA）
    """
    # 构建hybrid reference points
    if prev_bev is not None:
        prev_bev = torch.stack([prev_bev, bev_query], 1)  # [bs*2, len_bev, -1]
        hybird_ref_2d = torch.stack([shift_ref_2d, ref_2d], 1)  # [bs*2, len_bev, ...]
    
    for layer in self.layers:
        output = layer(bev_query, key, value, 
                      ref_2d=hybird_ref_2d,
                      ref_3d=ref_3d,
                      prev_bev=prev_bev, ...)
```

#### `custom_base_transformer_layer.py` - 自定义Transformer层基类
- **核心功能**: 支持灵活的attention配置
- **operation_order**: `('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')`
- **attn_cfgs**: 包含两个attention配置
  - `[0]`: TemporalSelfAttention (self_attn)
  - `[1]`: SpatialCrossAttention (cross_attn)

#### `temporal_self_attention.py` - 时间自注意力
- **功能**: 融合当前BEV和历史BEV
- **关键特性**:
  - 使用`num_bev_queue=2`（prev_bev和current_bev）
  - 使用多尺度可变形注意力（MSDeformableAttention）
  - 支持车辆运动补偿（通过shift参数）

**关键实现**:
```python
def forward(self, query, key, value, prev_bev=None, ref_2d=None, ...):
    """时间自注意力前向传播
    - query: 当前BEV查询
    - prev_bev: 历史BEV特征（已旋转对齐）
    - ref_2d: 2D参考点（BEV平面坐标）
    """
    # 构建value: [prev_bev, current_bev]
    if prev_bev is not None:
        value = torch.stack([prev_bev, query], dim=1)  # [bs, 2, N, C]
    else:
        value = torch.stack([query, query], dim=1)
    
    # 使用可变形注意力
    output = self.deform_attn(query, value, ref_2d, ...)
```

#### `spatial_cross_attention.py` - 空间交叉注意力
- **功能**: 从多相机图像特征中聚合信息到BEV
- **关键特性**:
  - 将BEV查询的3D参考点投影到各相机视图
  - 使用可变形注意力采样图像特征
  - 多视图融合（平均或求和）

**关键实现**:
```python
def forward(self, query, key, value, reference_points_cam=None, ...):
    """空间交叉注意力前向传播
    - query: BEV查询
    - reference_points_cam: 投影到各相机的参考点 [num_cam, B, N, D, 2]
    - bev_mask: 可见性掩码
    """
    # 逐相机处理
    for cam_id in range(num_cams):
        # 提取该相机的参考点和特征
        ref_points_cam = reference_points_cam[cam_id]
        feat_cam = value[cam_id]
        
        # 可变形注意力
        output_cam = self.deformable_attention(
            query, feat_cam, ref_points_cam, ...)
        outputs.append(output_cam)
    
    # 多视图融合
    output = torch.stack(outputs).mean(dim=0)
```

#### `decoder.py` - 检测解码器
- **功能**: 从BEV特征中解码出3D检测结果
- **结构**: 类似DETR decoder
  - Self-attention: 查询之间的交互
  - Cross-attention: 查询与BEV特征的交互
  - FFN: 前馈网络

## 数据流

### 训练流程

```
输入: img [B, T, V, C, H, W]
  ↓
1. 分离历史帧和当前帧
   - prev_img = img[:, :-1]  # [B, T-1, V, C, H, W]
   - img = img[:, -1]        # [B, V, C, H, W]
  ↓
2. 获取历史BEV (obtain_history_bev)
   for t in range(T-1):
     - 提取特征: img_feats = extract_feat(prev_img[:, t])
     - 获取BEV: prev_bev = transformer.get_bev_features(..., prev_bev)
   # 注意: 每帧使用前一帧的融合BEV作为prev_bev
  ↓
3. 提取当前帧特征
   - img_feats = extract_feat(img)  # List of [B, V, C, H_l, W_l]
  ↓
4. 完整检测流程
   - BEV特征: bev_embed = encoder(bev_queries, img_feats, prev_bev)
   - 检测结果: outputs = decoder(bev_embed, object_queries)
   - 损失计算: losses = loss(outputs, gt_bboxes_3d, gt_labels_3d)
```

### Encoder内部流程

```
BEV Queries [B, N, C]
  ↓
1. 处理prev_bev
   - 旋转对齐: prev_bev = rotate(prev_bev, can_bus[-1])
   - 构建hybrid: [prev_bev, current_bev] → [B*2, N, C]
  ↓
2. 生成参考点
   - ref_3d: 3D空间参考点 [B, N, D, 3] (用于SCA)
   - ref_2d: 2D BEV平面参考点 [B, N, 2] (用于TSA)
   - 投影: ref_3d → reference_points_cam [num_cam, B, N, D, 2]
  ↓
3. 逐层处理 (BEVFormerLayer × 6)
   for layer in layers:
     a. Temporal Self-Attention
        - 输入: query, prev_bev, ref_2d
        - 输出: 融合的BEV特征
     b. Spatial Cross-Attention
        - 输入: query, img_feats, reference_points_cam
        - 输出: 从图像特征聚合的BEV特征
     c. FFN
        - 输出: 最终BEV特征
  ↓
输出: BEV Embed [B, N, C]
```

## 关键设计要点

### 1. 迭代式历史BEV构建
- **原理**: 每帧的BEV融合前一帧的融合BEV，而不是独立处理
- **实现**: `obtain_history_bev`中循环处理，`prev_bev`不断更新
- **优势**: 保持时间连续性，累积历史信息

### 2. 车辆运动补偿
- **旋转**: 根据can_bus的yaw角度旋转prev_bev
- **平移**: 根据can_bus的x, y计算shift，调整参考点
- **实现位置**: `PerceptionTransformer.get_bev_features()`

### 3. 多尺度特征融合
- **FPN特征**: 4个尺度的特征图
- **层级嵌入**: `level_embeds`区分不同尺度
- **相机嵌入**: `cams_embeds`区分不同相机视图

### 4. 可变形注意力
- **TSA**: 2个level（prev_bev, current_bev），每个level采样多个点
- **SCA**: 4个level（FPN特征），每个level采样多个点
- **优势**: 高效处理大尺度特征图

## 配置文件结构

### 模型配置 (`configs/bevformer/bevformer_base.py`)
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

## 与原版对比要点

### 1. Encoder层结构
- **原版**: 每层包含TSA和SCA，在`BEVFormerLayer`中通过`attn_cfgs`配置
- **复现版**: 需要确保encoder的每一层都正确集成TSA和SCA

### 2. 历史BEV处理
- **原版**: `obtain_history_bev`迭代处理，每帧使用前一帧的融合BEV
- **复现版**: 已实现迭代逻辑，但需要确保与encoder的集成正确

### 3. 车辆运动补偿
- **原版**: 在`PerceptionTransformer.get_bev_features()`中处理旋转和平移
- **复现版**: 需要在合适的位置实现相同的补偿逻辑

### 4. 参考点生成
- **原版**: `BEVFormerEncoder.get_reference_points()`生成3D和2D参考点
- **复现版**: 需要确保参考点的格式和维度与原版一致

## 总结

BEVFormer的核心创新在于：
1. **时空融合**: 通过TSA融合历史BEV，通过SCA聚合多视图图像特征
2. **迭代式处理**: 历史BEV的迭代构建保持时间连续性
3. **运动补偿**: 通过can_bus信息补偿车辆运动
4. **可变形注意力**: 高效处理大尺度特征图

复现时需要特别注意：
- Encoder每层都要包含TSA和SCA
- 历史BEV的迭代构建逻辑
- 车辆运动补偿的实现位置
- 参考点的生成和投影

