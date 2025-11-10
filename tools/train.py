import os
import argparse
import os.path as osp
from mmcv import Config
from mmcv.runner import get_dist_info, init_dist
from mmdet3d.apis import set_random_seed, train_detector
from mmdet3d.models import build_model
from mmdet3d.datasets import build_dataset
import mmcv

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--resume-from', help='the checkpoint file to resume from')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch'],
        default='none',
        help='job launcher')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    
    # work_dir
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    
    if args.resume_from is not None and os.path.isfile(args.resume_from):
        cfg.resume_from = args.resume_from

    # init distributed env
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # create work_dir and dump config
    mmcv.mkdir_or_exist(os.path.abspath(cfg.work_dir))
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))

    # set random seeds
    set_random_seed(args.seed, deterministic=False)
    cfg.seed = args.seed

    # 数据集
    datasets = [build_dataset(cfg.data.train)]

    # 模型
    model = build_model(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model.init_weights()

    # 开训
    train_detector(model, datasets, cfg, distributed=distributed, validate=True)

if __name__ == '__main__':
    main()
