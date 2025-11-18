import os
import argparse
import os.path as osp
import copy
import warnings
import time
import random
import sys

# Add project root to Python path
tools_dir = osp.dirname(osp.abspath(__file__))
project_root = osp.dirname(tools_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import torch
import torch.distributed as dist
import logging
from mmcv import Config, DictAction
from mmcv.runner import (get_dist_info, init_dist, HOOKS, DistSamplerSeedHook,
                         EpochBasedRunner, Fp16OptimizerHook, OptimizerHook,
                         build_optimizer, build_runner)
from mmcv.utils import build_from_cfg
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel

from mmdet.datasets import build_dataset
from mmdet.utils import get_root_logger
from mmdet.apis import (set_random_seed)
from mmdet3d.utils import collect_env
import mmcv
# Import bevformer modules first to ensure all registrations happen
from bevformer import models  # noqa: F401 - Ensure models are registered
from bevformer.datasets.builder import build_dataloader
# Import build_model after bevformer modules are loaded
from mmdet3d.models import build_model


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--resume-from', help='the checkpoint file to resume from')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.')
    args = parser.parse_args()
    return args


def custom_train_detector(model,
                   dataset,
                   cfg,
                   distributed=False,
                   timestamp=None,
                   eval_model=None,
                   meta=None):
    """Custom training function for BEVFormer."""
    log_level = cfg.get('log_level', 'INFO')  # Default to INFO if not specified
    logger = get_root_logger(log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]

    if 'imgs_per_gpu' in cfg.data:
        logger.warning('"imgs_per_gpu" is deprecated in MMDet V2.0. '
                       'Please use "samples_per_gpu" instead')
        if 'samples_per_gpu' in cfg.data:
            logger.warning(
                f'Got "imgs_per_gpu"={cfg.data.imgs_per_gpu} and '
                f'"samples_per_gpu"={cfg.data.samples_per_gpu}, "imgs_per_gpu"'
                f'={cfg.data.imgs_per_gpu} is used in this experiments')
        else:
            logger.warning(
                'Automatically set "samples_per_gpu"="imgs_per_gpu"='
                f'{cfg.data.imgs_per_gpu} in this experiments')
        cfg.data.samples_per_gpu = cfg.data.imgs_per_gpu

    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed,
            shuffler_sampler=cfg.data.get('shuffler_sampler', None),
            nonshuffler_sampler=cfg.data.get('nonshuffler_sampler', None),
        ) for ds in dataset
    ]

    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
        if eval_model is not None:
            eval_model = MMDistributedDataParallel(
                eval_model.cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False,
                find_unused_parameters=find_unused_parameters)
    else:
        model = MMDataParallel(
            model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)
        if eval_model is not None:
            eval_model = MMDataParallel(
                eval_model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)

    if 'runner' not in cfg:
        cfg.runner = {
            'type': 'EpochBasedRunner',
            'max_epochs': cfg.total_epochs
        }
        warnings.warn(
            'config is now expected to have a `runner` section, '
            'please set `runner` in your config.', UserWarning)
    else:
        if 'total_epochs' in cfg:
            assert cfg.total_epochs == cfg.runner.max_epochs

    if eval_model is not None:
        runner = build_runner(
            cfg.runner,
            default_args=dict(
                model=model,
                eval_model=eval_model,
                optimizer=optimizer,
                work_dir=cfg.work_dir,
                logger=logger,
                meta=meta))
    else:
        runner = build_runner(
            cfg.runner,
            default_args=dict(
                model=model,
                optimizer=optimizer,
                work_dir=cfg.work_dir,
                logger=logger,
                meta=meta))

    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
    elif distributed and 'type' not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # register hooks
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config,
                                   cfg.get('momentum_config', None))

    if distributed:
        if isinstance(runner, EpochBasedRunner):
            runner.register_hook(DistSamplerSeedHook())

    # user-defined hooks
    if cfg.get('custom_hooks', None):
        custom_hooks = cfg.custom_hooks
        assert isinstance(custom_hooks, list), \
            f'custom_hooks expect list type, but got {type(custom_hooks)}'
        for hook_cfg in custom_hooks:
            assert isinstance(hook_cfg, dict), \
                'Each item in custom_hooks expects dict type, but got ' \
                f'{type(hook_cfg)}'
            hook_cfg = hook_cfg.copy()
            priority = hook_cfg.pop('priority', 'NORMAL')
            hook = build_from_cfg(hook_cfg, HOOKS)
            runner.register_hook(hook, priority=priority)

    if cfg.get('resume_from'):
        runner.resume(cfg.resume_from)
    elif cfg.get('load_from'):
        runner.load_checkpoint(cfg.load_from)
    workflow = cfg.get('workflow', [('train', 1)])  # Default to [('train', 1)] if not specified
    runner.run(data_loaders, workflow)


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # set tf32
    if cfg.get('close_tf32', False):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

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
        cfg.gpu_ids = range(1)
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # create work_dir and dump config
    mmcv.mkdir_or_exist(os.path.abspath(cfg.work_dir))
    # Fix for yapf version compatibility: copy original config file
    config_dump_path = osp.join(cfg.work_dir, osp.basename(args.config))
    try:
        cfg.dump(config_dump_path)
    except TypeError as e:
        # Fallback for yapf version incompatibility: copy original file
        if 'verify' in str(e):
            import shutil
            shutil.copy2(args.config, config_dump_path)
        else:
            raise

    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    log_level = cfg.get('log_level', 'INFO')  # Default to INFO if not specified
    logger = get_root_logger(log_level)
    # Set log file handler separately
    handler = logging.FileHandler(log_file, mode='w')
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info (with fallback for Windows without Visual C++)
    try:
        env_info_dict = collect_env()
        env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    except Exception as e:
        # Fallback: collect basic env info without C++ compiler check
        logger.warning(f'Failed to collect full environment info: {e}')
        env_info = f'PyTorch: {torch.__version__}\nCUDA available: {torch.cuda.is_available()}\nCUDA version: {torch.version.cuda if torch.cuda.is_available() else "N/A"}'
        if torch.cuda.is_available():
            env_info += f'\nGPU: {torch.cuda.get_device_name(0)}'
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info
    # Safe config text retrieval (avoid yapf verify issue)
    try:
        config_text = cfg.pretty_text
    except TypeError as e:
        if 'verify' in str(e):
            # Fallback: read original config file
            with open(args.config, 'r', encoding='utf-8') as f:
                config_text = f.read()
        else:
            raise
    meta['config'] = config_text

    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{config_text}')

    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed
    meta['exp_name'] = osp.basename(args.config)

    # 数据集
    logger.info('Building training dataset...')
    logger.info(f'Dataset config: type={cfg.data.train.get("type")}, data_root={cfg.data.train.get("data_root")}, ann_file={cfg.data.train.get("ann_file")}')
    
    # Check if data files exist
    ann_file = cfg.data.train.get("ann_file")
    if ann_file and os.path.exists(ann_file):
        file_size = os.path.getsize(ann_file) / (1024 * 1024)  # MB
        logger.info(f'Annotation file exists: {ann_file}, size: {file_size:.2f} MB')
    else:
        logger.warning(f'Annotation file not found: {ann_file}')
    
    logger.info('Starting dataset initialization (this may take a while for large datasets)...')
    sys.stdout.flush()  # Force flush output
    
    datasets = [build_dataset(cfg.data.train)]
    logger.info(f'Dataset built successfully. Dataset length: {len(datasets[0])}')
    sys.stdout.flush()

    # Handle workflow (train + val)
    logger.info('Checking workflow configuration...')
    workflow = cfg.get('workflow', [])
    logger.info(f'Workflow: {workflow}, length: {len(workflow)}')
    if len(workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        # in case we use a dataset wrapper
        if 'dataset' in cfg.data.train:
            val_dataset.pipeline = cfg.data.train.dataset.pipeline
        else:
            val_dataset.pipeline = cfg.data.train.pipeline
        # set test_mode=False here in deep copied config
        # which do not affect AP/AR calculation later
        val_dataset.test_mode = False
        datasets.append(build_dataset(val_dataset))

    # 模型
    logger.info('Building model...')
    logger.info(f'Model config type: {cfg.model.get("type")}')
    logger.info(f'Model config keys: {list(cfg.model.keys())[:10]}...')  # Show first 10 keys
    sys.stdout.flush()
    
    logger.info('Calling build_model function...')
    logger.info(f'train_cfg: {cfg.get("train_cfg")}')
    logger.info(f'test_cfg: {cfg.get("test_cfg")}')
    sys.stdout.flush()
    
    # Add timeout or detailed logging
    import traceback
    logger.info('About to call build_model, stack trace:')
    logger.info(''.join(traceback.format_stack()[-3:-1]))
    sys.stdout.flush()
    
    # Check CUDA availability before building model
    logger.info(f'CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        logger.info(f'CUDA device count: {torch.cuda.device_count()}')
        logger.info(f'Current CUDA device: {torch.cuda.current_device()}')
        logger.info(f'CUDA device name: {torch.cuda.get_device_name(0)}')
    sys.stdout.flush()
    
    try:
        logger.info('Entering build_model...')
        logger.info('This may take a while - building ResNet101, FPN, and BEVFormerHead...')
        logger.info('If this hangs, it might be building img_backbone (ResNet101) or img_neck (FPN)...')
        logger.info('Note: ResNet101 with DCNv2 can take 30-60 seconds to build on first run')
        sys.stdout.flush()
        
        # Try to add periodic heartbeat to see if process is alive
        import threading
        heartbeat_active = [True]
        
        def heartbeat():
            count = 0
            while heartbeat_active[0]:
                time.sleep(5)  # Every 5 seconds
                count += 1
                if count <= 24:  # Log for up to 120 seconds
                    logger.info(f'[Heartbeat] Still building model... ({count * 5}s elapsed)')
                    sys.stdout.flush()
        
        heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
        heartbeat_thread.start()
        
        # Measure build time
        start_time = time.time()
        
        model = build_model(
            cfg.model,
            train_cfg=cfg.get('train_cfg'),
            test_cfg=cfg.get('test_cfg'))
        
        heartbeat_active[0] = False
        elapsed = time.time() - start_time
        logger.info(f'build_model returned successfully (took {elapsed:.2f} seconds)')
        
    except Exception as e:
        heartbeat_active[0] = False
        logger.error(f'Exception in build_model: {type(e).__name__}: {e}')
        logger.error('Full traceback:', exc_info=True)
        raise
    sys.stdout.flush()
    logger.info('Model built. Initializing weights...')
    sys.stdout.flush()
    model.init_weights()
    logger.info('Model weights initialized.')

    logger.info(f'Model:\n{model}')

    # Set model classes
    model.CLASSES = datasets[0].CLASSES

    # Set checkpoint meta if available
    if cfg.checkpoint_config is not None:
        cfg.checkpoint_config.meta = dict(
            config=config_text,
            CLASSES=datasets[0].CLASSES)

    # 开训
    custom_train_detector(
        model,
        datasets,
        cfg,
        distributed=distributed,
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    main()
