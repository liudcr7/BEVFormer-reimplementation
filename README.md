# BEVFormer-from-scratch (OpenMMLab 1.x)

This repo aims to re-implement BEVFormer from scratch, aligned with the original design.
- Config: `configs/bevformer_base.py` (provided by user)
- Data: directly use prepared `.pkl` (no create_data step)
- Framework: MMCV 1.x / MMDet 2.x / MMDet3D 1.x

## Quick Start
```bash
conda env create -f env/conda_env_cuda11.yml
conda activate bevformer
python tools/train.py configs/bevformer_base.py --work-dir work_dirs/bevformer_base


conda create -n open-mmlab python=3.8 -y
conda activate open-mmlab

pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0 -f https://download.pytorch.org/whl/cu113/torch_stable.html

pip install -U openmim
mim install mmcv-full==1.6.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10/index.html

pip install mmdet==2.28.2
pip install mmsegmentation==0.30.0

pip install mmdet==2.28.2 mmsegmentation==0.30.0 mmdet3d==1.0.0rc6


pip install einops fvcore "iopath==0.1.9" "timm==0.6.13"   "typing-extensions>=4.13.0" ipython==8.12   numpy==1.23.5 "numba==0.53.0" pandas==1.5.3   "scikit-image==0.19.3" matplotlib==3.6.3 seaborn==0.12.2

pip install "matplotlib==3.5.3"

python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

export PYTHONPATH=$PYTHONPATH:"./"

python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes --version v1.0-mini --canbus ./data

