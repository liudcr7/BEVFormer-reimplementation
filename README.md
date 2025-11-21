# BEVFormer (OpenMMLab 1.x)

This repo aims to re-implement BEVFormer from scratch, aligned with the original design.
- Config: `configs/config.py`
- Framework: MMCV 1.6.0 / MMDet 2.28.2 / MMDet3D 1.0.0rc6

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
shit
