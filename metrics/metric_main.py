# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import time
import json
from pathlib import Path
from typing import Optional, Callable

import torch

import dnnlib
from metrics import metric_utils
from metrics import frechet_inception_distance
from metrics import precision_recall
from metrics import clip_score

_metric_dict = dict() # name => fn

def register_metric(fn: Callable) -> Callable:
    assert callable(fn)
    _metric_dict[fn.__name__] = fn
    return fn

def is_valid_metric(metric: str) -> bool:
    return metric in _metric_dict

def list_valid_metrics() -> list[str]:
    return list(_metric_dict.keys())

def calc_metric(metric: str, **kwargs) -> dnnlib.EasyDict: # See metric_utils.MetricOptions for the full list of arguments.
    assert is_valid_metric(metric)
    opts = metric_utils.MetricOptions(**kwargs)

    # Calculate.
    start_time = time.time()
    results = _metric_dict[metric](opts)
    total_time = time.time() - start_time

    # Broadcast results.
    for key, value in list(results.items()):
        if opts.num_gpus > 1:
            value = torch.as_tensor(value, dtype=torch.float64, device=opts.device)
            torch.distributed.broadcast(tensor=value, src=0)
            value = float(value.cpu())
        results[key] = value

    # Decorate with metadata.
    return dnnlib.EasyDict(
        results         = dnnlib.EasyDict(results),
        metric          = metric,
        total_time      = total_time,
        total_time_str  = dnnlib.util.format_time(total_time),
        num_gpus        = opts.num_gpus,
    )

def report_metric(result_dict: dict, run_dir: Optional[str] = None, snapshot_pkl: Optional[str] = None) -> None:
    metric = result_dict['metric']
    assert is_valid_metric(metric)
    if run_dir is not None and snapshot_pkl is not None:
        snapshot_pkl = os.path.relpath(snapshot_pkl, run_dir)

    jsonl_line = json.dumps(dict(result_dict, snapshot_pkl=snapshot_pkl, timestamp=time.time()))
    print(jsonl_line)
    if run_dir is not None and os.path.isdir(run_dir):
        with open(os.path.join(run_dir, f'metric-{metric}.jsonl'), 'at') as f:
            f.write(jsonl_line + '\n')

def get_coco_path(original_path: str) -> str:
    # Check if coco path was already provided.
    if Path(original_path).stem == 'coco_val256':
        return original_path
    # Check if coco.zip is in data folder of the current dataset.
    elif (Path(original_path).parent / 'coco_val256.zip').exists():
        return str(Path(original_path).parent / 'coco_val256.zip')
    # Check if coco.zip in ENV.
    else:
        path = ''
        if 'COCOPATH' in os.environ: 
            path = os.environ["COCOPATH"]
            
        if Path(path).stem == 'coco_val256':
            return path
        else:
            raise ValueError(f'Did not find coco_val256. $COCOPATH: {path}')

### General Metrics

@register_metric
def fid50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    is_imfolder = opts.dataset_kwargs.class_name == 'training.data_zip.ImageFolderDataset'
    assert is_imfolder, f'Calculating metrics on {opts.dataset_kwargs.class_name} are not supported'
    fid = frechet_inception_distance.compute_fid(opts, max_real=None, num_gen=50000)
    return dict(fid50k_full=fid)

@register_metric
def fid10k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    is_imfolder = opts.dataset_kwargs.class_name == 'training.data_zip.ImageFolderDataset'
    assert is_imfolder, f'Calculating metrics on {opts.dataset_kwargs.class_name} are not supported'
    fid = frechet_inception_distance.compute_fid(opts, max_real=None, num_gen=10000)
    return dict(fid10k_full=fid)

@register_metric
def cs10k(opts):
    assert opts.G.c_dim > 1, 'CLIP score only works for conditional generators.'
    opts.dataset_kwargs.update(
        class_name='training.data_zip.ImageFolderDataset',
        max_size=None, xflip=False,
    )
    cs = clip_score.compute_clip_score(opts, num_gen=10000)
    return dict(cs=cs)

@register_metric
def pr50k3_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    is_imfolder = opts.dataset_kwargs.class_name == 'training.data_zip.ImageFolderDataset'
    assert is_imfolder, f'Calculating metrics on {opts.dataset_kwargs.class_name} are not supported'
    precision, recall = precision_recall.compute_pr(opts, max_real=200000, num_gen=50000, nhood_size=3, row_batch_size=10000, col_batch_size=10000)
    return dict(pr50k3_full_precision=precision, pr50k3_full_recall=recall)

### Zero-shot COCO Metrics

@register_metric
def fid30k_coco64(opts):
    coco_path = get_coco_path(opts.dataset_kwargs.path)
    opts.dataset_kwargs.update(
        class_name="training.data_zip.ImageFolderDataset",
        path=coco_path, resolution=64, max_size=None, xflip=False,
    )
    fid = frechet_inception_distance.compute_fid(opts, max_real=None, num_gen=30000)
    return dict(fid30k_full_coco_val=fid)

@register_metric
def fid30k_coco256(opts):
    coco_path = get_coco_path(opts.dataset_kwargs.path)
    opts.dataset_kwargs.update(
        class_name="training.data_zip.ImageFolderDataset",
        path=coco_path, resolution=256, max_size=None, xflip=False,
    )
    fid = frechet_inception_distance.compute_fid(opts, max_real=None, num_gen=30000)
    return dict(fid30k_full_coco_val=fid)

@register_metric
def cs10k_coco(opts):
    coco_path = get_coco_path(opts.dataset_kwargs.path)
    assert opts.G.c_dim > 1, 'CLIP score only works for conditional generators.'
    opts.dataset_kwargs.update(
        class_name='training.data_zip.ImageFolderDataset',
        path=coco_path, max_size=None, xflip=False,
    )
    cs = clip_score.compute_clip_score(opts, num_gen=30000)
    return dict(cs=cs)
