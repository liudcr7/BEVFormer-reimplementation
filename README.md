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
