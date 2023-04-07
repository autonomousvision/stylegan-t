# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Calculate quality metrics for previous training run or pretrained network pickle."""

import os
import json
import copy

import torch
import dill
import click

import dnnlib
from metrics import metric_main
from metrics import metric_utils
from torch_utils import misc
from torch_utils import custom_ops
from torch_utils import distributed as dist
from torch_utils.ops import conv2d_gradfix


def parse_comma_separated_list(s):
    if isinstance(s, list):
        return s
    if s is None or s.lower() == 'none' or s == '':
        return []
    return s.split(',')


@click.command("cli", context_settings={'show_default': True})
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename or URL', type=str, required=True)
@click.option('--metrics',                help='Quality metrics',                type=parse_comma_separated_list, default='fid50k_full')
@click.option('--data',                   help='Dataset to evaluate against',    type=str)
@click.option('--mirror',                 help='Enable dataset x-flips',         type=bool)
@click.option('--truncation',             help='Truncation',                     type=float, default=1.0)
def calc_metrics(
    ctx,
    network_pkl: str,
    metrics: list,
    data: str,
    mirror: bool,
    truncation: float,
):
    """Calculate quality metrics for previous training run or pretrained network pickle.

    Examples:

    \b
    # Previous training run: look up options automatically, save result to JSONL file.
    python calc_metrics.py --metrics=cs10k,fid50k_full \\
        --network=~/training-runs/00000-mydataset@512-custom-gpus1-b4-bgpu2/network-snapshot-000000.pkl

    \b
    # Pre-trained network pickle: specify dataset explicitly, print result to stdout.
    python calc_metrics.py --metrics=fid50k_full --data=~/datasets/mydataset.zip --mirror=1 \\
        --network=~/training-runs/00000-mydataset@512-custom-gpus1-b4-bgpu2/network-snapshot-000000.pkl

    \b
    General metrics:
      fid50k_full      Frechet inception distance against the full dataset (50k generated samples).
      fid10k_full      Frechet inception distance against the full dataset (10k generated samples).
      cs10k            Clip score (10k generated samples).
      pr50k3_full      Precision and recall againt the full dataset (50k generated samples, neighborhood size=3).

    \b
    Zero-shot COCO metrics:
      fid30k_coco64    Frechet inception distance against the COCO validation set (30k generated samples).
      fid30k_coco256   Frechet inception distance against the COCO validation set (30k generated samples).
      cs10k_coco       Clip score on the COCO validation set (10k generated samples).
    """

    # Init distributed
    torch.multiprocessing.set_start_method('spawn')
    dist.init()
    device = torch.device('cuda')

    # Validate arguments.
    G_kwargs=dnnlib.EasyDict(truncation_psi=truncation)

    if not all(metric_main.is_valid_metric(metric) for metric in metrics):
        err = ['--metrics can only contain the following values:'] + metric_main.list_valid_metrics()
        ctx.fail('\n'.join(err))

    # Load network.
    if not dnnlib.util.is_url(network_pkl, allow_file_urls=True) and not os.path.isfile(network_pkl):
        ctx.fail('--network must point to a file or URL')

    dist.print0(f'Loading network from "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl, verbose=True) as f:
        network_dict = dill.load(f)
        G = network_dict['G_ema'] # subclass of torch.nn.Module

    # Initialize dataset options.
    if data is not None:
        dataset_kwargs = dnnlib.EasyDict(class_name='training.data_zip.ImageFolderDataset', path=data)
    elif network_dict.get('training_set_kwargs') is not None:
        dataset_kwargs = dnnlib.EasyDict(network_dict['training_set_kwargs'])
    else:
        ctx.fail('Could not look up dataset options; please specify --data')

    # Finalize dataset options.
    dataset_kwargs.resolution = G.img_resolution
    dataset_kwargs.use_labels = (G.c_dim != 0)
    if mirror is not None:
        dataset_kwargs.xflip = mirror

    # Print dataset options.
    dist.print0('Dataset options:')
    dist.print0(json.dumps(dataset_kwargs, indent=2))

    # Locate run dir.
    run_dir = None
    if os.path.isfile(network_pkl):
        pkl_dir = os.path.dirname(network_pkl)
        if os.path.isfile(os.path.join(pkl_dir, 'training_options.json')):
            run_dir = pkl_dir

    # Launch processes.
    dist.print0('Launching processes...')
    dnnlib.util.Logger(should_flush=True)
    if dist.get_rank() != 0:
        custom_ops.verbosity = 'none'

    # Configure torch.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    conv2d_gradfix.enabled = True

    # Print network summary.
    G = copy.deepcopy(G).eval().requires_grad_(False).to(device)
    if dist.get_rank() == 0:
        z = torch.empty([1, G.z_dim], device=device)
        c = torch.empty([1, G.c_dim], device=device)
        misc.print_module_summary(G, [z, c])

    # Calculate each metric.
    for metric in metrics:
        dist.print0(f'Calculating {metric}...')

        progress = metric_utils.ProgressMonitor(verbose=True)
        result_dict = metric_main.calc_metric(metric=metric, G=G, G_kwargs=G_kwargs, dataset_kwargs=dataset_kwargs,
            num_gpus=dist.get_world_size(), rank=dist.get_rank(), device=device, progress=progress)

        if dist.get_rank() == 0:
            metric_main.report_metric(result_dict, run_dir=run_dir, snapshot_pkl=network_pkl)
        dist.print0()

    # Done.
    dist.print0('Exiting...')


if __name__ == "__main__":
    calc_metrics() # pylint: disable=no-value-for-parameter
