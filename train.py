# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
Train a GAN using the techniques described in the paper
"StyleGAN-T: Unlocking the Power of GANs for Fast Large-Scale Text-to-Image Synthesis".
"""

import os
import re
import json
from pathlib import Path
from typing import Union
from glob import glob

import torch
import click

import dnnlib
from training import training_loop
from metrics import metric_main
from torch_utils import distributed as dist
from torch_utils import custom_ops


def parse_comma_separated_list(s: Union[None, str, list]) -> Union[list, str]:
    if isinstance(s, list):
        return s
    if s is None or s.lower() == 'none' or s == '':
        return []
    return s.split(',')


def is_power_of_two(n: int) -> bool:
    return (n != 0) and (n & (n-1) == 0)


def init_dataset_kwargs(data: str, resolution: int) -> dnnlib.EasyDict:
    d_kwargs = dnnlib.EasyDict(path=data, xflip=False, use_labels=True)
    is_wds = len(glob(f'{data}/**/*.tar')) > 0  # check if files are tars, then it's a webdataset
    if is_wds:
        assert resolution, "Provide desired resolution when training on webdatasets."
        d_kwargs.class_name = 'training.data_wds.WdsWrapper'
    else:
        d_kwargs.class_name = 'training.data_zip.ImageFolderDataset'
        dataset_obj = dnnlib.util.construct_class_by_name(**d_kwargs) # Subclass of training.dataset.Dataset.
        assert resolution <= dataset_obj._raw_shape[-1], f"Native dataset resolution is smaller than {resolution}"
    assert is_power_of_two(resolution)
    d_kwargs.resolution = resolution
    return d_kwargs


@click.command("cli", context_settings={'show_default': True})
# Required.
@click.option('--outdir',          help='Where to save the results',        type=str, required=True)
@click.option('--data',            help='Training data',                    type=str, required=True)
@click.option('--img-resolution',  help='Resolution for webdatasets',       type=click.IntRange(min=8), required=True)
@click.option('--batch',           help='Total batch size',                 type=click.IntRange(min=1), required=True)
@click.option('--batch-gpu',       help='Limit batch size per GPU',         type=click.IntRange(min=1), default=8)
# G architecture args
@click.option('--cfg',             help='Base config.',                     type=click.Choice(['custom', 'lite', 'full']), default='custom')
@click.option('--cbase',           help='Capacity multiplier',              type=click.IntRange(min=1), default=32768)
@click.option('--cmax',            help='Max. feature maps',                type=click.IntRange(min=1), default=512)
@click.option('--res-blocks',      help='Number of residual blocks',        type=click.IntRange(min=1), default=2)
# Resuming.
@click.option('--resume',          help='Resume from given network pickle', type=str)
@click.option('--resume-kimg',     help='Resume from given kimg',           type=click.IntRange(min=0), default=0)
# Training params.
@click.option('--train-mode',      help='Which layers to train',            type=click.Choice(['all', 'text_encoder', 'freeze64']), default='all')
@click.option('--clip-weight',     help='Loss weight for clip loss',        type=float, default=0)
@click.option('--blur-init',       help='Init blur width',                  type=click.IntRange(min=0), default=32,)
@click.option('--blur-fade-kimg',  help='Discriminator blur duration',      type=click.IntRange(min=0), default=1000)
# Misc settings.
@click.option('--suffix',          help='suffix of result dirname',         type=str, default='')
@click.option('--metrics',         help='Quality metrics',                  type=parse_comma_separated_list, default=[])
@click.option('--kimg',            help='Total training duration',          type=click.IntRange(min=1), default=25000)
@click.option('--tick',            help='How often to print progress',      type=click.IntRange(min=1), default=4)
@click.option('--snap',            help='How often to save snapshots',      type=click.IntRange(min=1), default=100)
@click.option('--seed',            help='Random seed',                      type=click.IntRange(min=0), default=0)
@click.option('--fp32',            help='Disable mixed-precision',          type=bool, default=False)
@click.option('--nobench',         help='Disable cuDNN benchmarking',       type=bool, default=False)
@click.option('--workers',         help='DataLoader worker processes',      type=click.IntRange(min=1), default=3)
@click.option('--dry-run',         help='Print training options and exit',  type=bool, is_flag=True)
def main(**kwargs) -> None:
    # Initialize config.
    torch.multiprocessing.set_start_method('spawn')
    dist.init()
    opts = dnnlib.EasyDict(kwargs)
    c = dnnlib.EasyDict()

    # Networks.
    c.D_kwargs = dnnlib.EasyDict(class_name='networks.discriminator.ProjectedDiscriminator')
    c.G_kwargs = dnnlib.EasyDict(class_name='networks.generator.Generator', z_dim=64)
    c.G_kwargs.train_mode = opts.train_mode

    # Synthesis settings.
    cfg_synthesis = {
        'full': dnnlib.EasyDict(channel_base=65536, channel_max=2048, num_res_blocks=3),
        'lite': dnnlib.EasyDict(channel_base=32768, channel_max=512, num_res_blocks=2),
        'custom': dnnlib.EasyDict(channel_base=opts.cbase, channel_max=opts.cmax, num_res_blocks=opts.res_blocks),
    }
    c.G_kwargs.synthesis_kwargs = cfg_synthesis[opts.cfg]
    c.G_kwargs.synthesis_kwargs.architecture = 'skip'

    # Optimizer.
    c.G_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', betas=[0, 0.99], eps=1e-8, lr=0.002)
    c.D_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', betas=[0, 0.99], eps=1e-8, lr=0.002)
    if c.G_kwargs.train_mode == 'text_encoder':
        c.G_opt_kwargs.lr = 3e-6

    # Loss.
    c.loss_kwargs = dnnlib.EasyDict(class_name='training.loss.ProjectedGANLoss')
    c.loss_kwargs.blur_init_sigma = opts.blur_init
    c.loss_kwargs.blur_fade_kimg = opts.blur_fade_kimg
    c.loss_kwargs.clip_weight = opts.clip_weight

    # Data.
    c.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, num_workers=3, prefetch_factor=2)
    c.training_set_kwargs = init_dataset_kwargs(data=opts.data, resolution=opts.img_resolution)
    c.G_kwargs.img_resolution = c.training_set_kwargs.resolution
    desc = f'{Path(c.training_set_kwargs.path).stem}@{c.training_set_kwargs.resolution}-{opts.cfg}-'

    # Logging.
    c.random_seed = c.training_set_kwargs.random_seed = opts.seed
    c.image_snapshot_ticks = c.network_snapshot_ticks = opts.snap
    c.metrics = opts.metrics
    c.total_kimg = opts.kimg
    c.kimg_per_tick = opts.tick

    # GPUs and batch size.
    c.batch_size = opts.batch
    c.batch_gpu = opts.batch_gpu
    c.ema_kimg = c.batch_size * 10 / 32
    desc += f'gpus{dist.get_world_size():d}-b{c.batch_size}-bgpu{c.batch_gpu}'

    # Sanity checks.
    if c.batch_size % dist.get_world_size() != 0:
        raise click.ClickException('--batch must be a multiple of --gpus')
    if c.batch_size % (dist.get_world_size() * c.batch_gpu) != 0:
        raise click.ClickException('--batch must be a multiple of --gpus times --batch-gpu')
    if any(not metric_main.is_valid_metric(metric) for metric in c.metrics):
        err = ['--metrics can only contain the following values:'] + metric_main.list_valid_metrics()
        raise click.ClickException('\n'.join(err))

    # Resume.
    if opts.resume is not None:
        c.resume_pkl = opts.resume
        c.resume_kimg = opts.resume_kimg
        c.ema_rampup = None  # Disable EMA rampup.

    # Performance-related toggles.
    if opts.fp32:
        c.G_kwargs.num_fp16_res = 0
        c.G_kwargs.conv_clamp = None
    if opts.nobench:
        c.cudnn_benchmark = False

    if opts.suffix:
        desc += f'-{opts.suffix}'

    # Pick output directory.
    if dist.get_rank() != 0:
        c.run_dir = None
    else:
        prev_run_dirs = []
        if os.path.isdir(opts.outdir):
            prev_run_dirs = [x for x in os.listdir(opts.outdir) if os.path.isdir(os.path.join(opts.outdir, x))]
        prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
        prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
        cur_run_id = max(prev_run_ids, default=-1) + 1
        c.run_dir = os.path.join(opts.outdir, f'{cur_run_id:05d}-{desc}')
        assert not os.path.exists(c.run_dir)

    # Print options.
    dist.print0()
    dist.print0('Training options:')
    dist.print0(json.dumps(c, indent=2))
    dist.print0()
    dist.print0(f'Output directory:    {c.run_dir}')
    dist.print0(f'Number of GPUs:      {dist.get_world_size()}')
    dist.print0(f'Batch size:          {c.batch_size} images')
    dist.print0(f'Training duration:   {c.total_kimg} kimg')
    dist.print0(f'Dataset path:        {c.training_set_kwargs.path}')
    dist.print0(f'Dataset resolution:  {c.training_set_kwargs.resolution}')
    dist.print0(f'Dataset labels:      {c.training_set_kwargs.use_labels}')
    dist.print0(f'Dataset x-flips:     {c.training_set_kwargs.xflip}')
    dist.print0()

    # Dry run?
    if opts.dry_run:
        dist.print0('Dry run; exiting.')
        return

    # Create output directory.
    dist.print0('Creating output directory...')
    if dist.get_rank() == 0:
        os.makedirs(c.run_dir, exist_ok=True)
        with open(os.path.join(c.run_dir, 'training_options.json'), 'wt+') as f:
            json.dump(c, f, indent=2)

        dnnlib.util.Logger(file_name=os.path.join(c.run_dir, 'log.txt'), file_mode='a', should_flush=True)
    else:
        custom_ops.verbosity = 'none'

    # Train.
    training_loop.training_loop(**c)

if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
