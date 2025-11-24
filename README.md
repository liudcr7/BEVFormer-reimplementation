# BEVFormer (OpenMMLab 1.x)

This repo aims to re-implement BEVFormer model part from scratch, aligned with the original design.
- Config: `configs/config.py`
- Framework: MMCV 1.6.0 / MMDet 2.28.2 / MMDet3D 1.0.0rc6 （Must be the same!!!）

## Quick Start
```bash

ENV for A100
conda create -n open-mmlab python=3.8 -y
conda activate open-mmlab

pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0 -f https://download.pytorch.org/whl/cu113/torch_stable.html

pip install -U openmim
mim install mmcv-full==1.6.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10/index.html

pip install mmdet==2.28.2
pip install mmsegmentation==0.30.0

pip install mmdet==2.28.2 mmsegmentation==0.30.0 mmdet3d==1.0.0rc6


pip install einops fvcore "iopath==0.1.9" "timm==0.6.13"   "typing-extensions>=4.13.0" ipython==8.12   numpy==1.23.5 "numba==0.53.0" pandas==1.5.3   "scikit-image==0.19.3" matplotlib==3.5.3 seaborn==0.12.2

pip install "setuptools==59.5.0" 

python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

export PYTHONPATH=$PYTHONPATH:"./"

python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes --version v1.0-mini --canbus ./data

python tools/train.py configs/config.py
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29500 tools/train.py configs/config.py --launcher pytorch

python tools/eval.py configs/config.py work_dirs/xxx/epoch_X.pth --eval bbox

ENV for H800(sm90)
# Similar setup as A100, but may require different CUDA version for sm_90 architecture
```

## Implementation Details

### Memory Consumption

This implementation consumes more GPU memory compared to the original BEVFormer, primarily due to the following reasons:

1. **Pure PyTorch Implementation without CUDA Extensions**
   - Multi-Scale Deformable Attention is implemented in pure PyTorch using `grid_sample` for bilinear interpolation
   - While the pure PyTorch implementation is more portable, it consumes more memory and runs slower compared to CUDA kernel implementations

2. **FP16 Mixed Precision Training Not Enabled**
   - Although we built FP16 support (`@auto_fp16` decorators), it is disabled by default (`fp16_enabled = False`)
   - All computations use FP32 precision, which approximately doubles memory consumption compared to FP16 training

3. **Actual Memory Usage**
   - **A100 (40GB)**: Can only run with BEV size of 120×120
   - **H800 (80GB)**: Requires approximately 55512MiB (~54GB) memory to run with BEV size of 200×200

### Training 

1. **V1.0-mini**
   - Eval command:
     ```bash
     python tools/eval.py configs/config120.py work_dirs/config120/epoch_24.pth --eval bbox
     ```
   - Due to the small dataset size, model performance is significantly worse compared to training on the full dataset
   （blue is the prediction result, green is the groundtruth）
   ![BEV Visualization](result/bev120x120-bs1-epoch24-mini/img/a98fba72bde9433fb882032d18aedb2e_bev.png)
   ![Camera View](result/bev120x120-bs1-epoch24-mini/img/a98fba72bde9433fb882032d18aedb2e_camera.png)


2. **V1.0-trainval**
   - Eval command:
     ```bash
     python tools/eval.py configs/config200.py work_dirs/config200/epoch_7.pth --eval bbox
     ```
   - The original paper trains for 24 epochs on the full dataset
   - Limited by computational resources, this implementation only train for 7 epochs：
   ![Learning rate](result/bev200x200-bs1-epoch7/lr.png)
   ![Loss](result/bev200x200-bs1-epoch7/loss.png)  
   - The result:

     | mAP | mATE | mASE | mAOE | mAVE | mAAE | NDS |
     |-----|------|------|------|------|------|-----|
     | 0.2408 | 1.1082 | 0.7151 | 1.5707 | 0.5132 | 0.2109 | 0.2765 |

     | Class | AP | ATE | ASE | AOE | AVE | AAE |
     |-------|----|-----|-----|-----|-----|-----|
     | car | 0.361 | 1.115 | 0.751 | 1.618 | 0.399 | 0.201 |
     | truck | 0.211 | 1.209 | 0.792 | 1.624 | 0.498 | 0.221 |
     | bus | 0.265 | 1.116 | 0.860 | 1.582 | 1.173 | 0.363 |
     | trailer | 0.073 | 1.147 | 0.845 | 1.634 | 0.367 | 0.189 |
     | construction_vehicle | 0.049 | 1.331 | 0.726 | 1.490 | 0.146 | 0.317 |
     | pedestrian | 0.330 | 1.050 | 0.332 | 1.602 | 0.433 | 0.193 |
     | motorcycle | 0.240 | 1.046 | 0.801 | 1.541 | 0.813 | 0.191 |
     | bicycle | 0.223 | 1.074 | 0.816 | 1.735 | 0.276 | 0.013 |
     | traffic_cone | 0.330 | 1.009 | 0.338 | nan | nan | nan |
     | barrier | 0.326 | 0.986 | 0.890 | 1.310 | nan | nan |

   - And this is one example:
   ![BEV Visualization](result/bev200x200-bs1-epoch7/img/f7d75d25c86941f3aecfed9efea1a3e3_bev.png)
   ![Camera View](result/bev200x200-bs1-epoch7/img/f7d75d25c86941f3aecfed9efea1a3e3_camera.png)
   We can see that the model is quite precise in x and y, but z still need to learn.


