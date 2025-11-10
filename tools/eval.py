import argparse
import os
from mmcv import Config
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmdet3d.apis import single_gpu_test, multi_gpu_test
from mmdet3d.datasets import build_dataset, build_dataloader
from mmdet3d.models import build_model
from mmdet.apis import set_random_seed
import torch
import mmcv

def parse_args():
    parser = argparse.ArgumentParser(description='Test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        default=['bbox'],
        help='evaluation metrics, e.g., "bbox"')
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
    cfg.model.pretrained = None
    
    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])

    # init distributed env
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # set random seeds
    set_random_seed(0, deterministic=False)

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False
    )

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    
    # old versions did not save class info in checkpoints
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=True)

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)

        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # remove EvalHook args
            for key in ['interval', 'tmpdir', 'start', 'gpu_collect', 'save_best', 'rule']:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval))
            print(dataset.evaluate(outputs, **eval_kwargs))

if __name__ == '__main__':
    main()
