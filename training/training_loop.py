# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Main training loop."""

import os
import math
import time
import copy
import json
import PIL.Image
from typing import Union, Iterator, Optional, Any

import dill
import psutil
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils.tensorboard as tensorboard

import dnnlib
from torch_utils import training_stats
from torch_utils import misc
from torch_utils import distributed as dist
from torch_utils.ops import conv2d_gradfix
from metrics import metric_main
from training.data_wds import wds_dataloader
from docs.prompts import PROMPTS_TRAINING


# Visualization
# ------------------------------------------------------------------------------------------

def setup_snapshot_image_grid(
    training_set: Any,
    random_seed: int = 0,
    gw: Optional[int] = None,
    gh: Optional[int] = None,
) -> tuple[tuple[int,int], np.ndarray, np.ndarray]:

    rnd = np.random.RandomState(random_seed)
    if gw is None:
        gw = np.clip(7680 // training_set.image_shape[2], 7, 32)
    if gh is None:
        gh = np.clip(4320 // training_set.image_shape[1], 4, 32)

    # No labels => show random subset of training samples.
    if not training_set.has_labels:
        all_indices = list(range(len(training_set)))
        rnd.shuffle(all_indices)
        grid_indices = [all_indices[i % len(all_indices)] for i in range(gw * gh)]

    elif training_set.labels_are_text:
        all_indices = list(range(len(training_set)))
        grid_indices = [all_indices[i % len(all_indices)] for i in range(gw * gh)]

    else:
        # Group training samples by label.
        label_groups = dict() # label => [idx, ...]
        for idx in range(len(training_set)):
            label = tuple(training_set.get_details(idx).raw_label.flat[::-1])
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(idx)

        # Reorder.
        label_order = sorted(label_groups.keys())
        for label in label_order:
            rnd.shuffle(label_groups[label])

        # Organize into grid.
        grid_indices = []
        for y in range(gh):
            label = label_order[y % len(label_order)]
            indices = label_groups[label]
            grid_indices += [indices[x % len(indices)] for x in range(gw)]
            label_groups[label] = [indices[(i + gw) % len(indices)] for i in range(len(indices))]

    # Load data.
    images, labels = zip(*[training_set[i] for i in grid_indices])
    return (gw, gh), np.stack(images), np.stack(labels)


def save_image_grid(
    img: torch.Tensor,
    drange: tuple[int, int],
    grid_size: tuple[int, int],
    fname: str = '',
) -> Optional[np.ndarray]:
    """Build image grid, save if fname is given"""
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, H, W = img.shape
    img = img.reshape([gh, gw, C, H, W])
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape([gh * H, gw * W, C])

    # Save or return.
    if fname:
        if C == 3:
            PIL.Image.fromarray(img, 'RGB').save(fname)
        else:
            PIL.Image.fromarray(img[:, :, 0], 'L').save(fname)
    else:
        return img


def save_prompts_dict(
    prompts_dict: dict,
    path: str,
    figsize: tuple[int, int] = (100,50),
) -> None:
    _, axs = plt.subplots(len(prompts_dict), 1, constrained_layout=True, figsize=figsize)
    for i, (prompt, imgs) in enumerate(prompts_dict.items()):
        axs[i].set_title(prompt)
        axs[i].imshow(imgs);
        axs[i].axis('off')
    plt.savefig(path, bbox_inches='tight')


def save_samples(
    G_ema: nn.Module,
    run_dir: str,
    device: torch.device,
    suffix: str,
    prompts: list[str] = PROMPTS_TRAINING,
    n_samples: int = 5,
) -> None:
    save_path = os.path.join(run_dir, f"samples_{suffix}.png")

    if G_ema.c_dim > 0:
        z = np.random.RandomState(0).randn(n_samples, G_ema.z_dim);
        z = torch.from_numpy(z).float().to(device)

        prompts_results = dict()
        for prompt in prompts:
            ws = [
                G_ema.mapping(z, [prompt]*len(z), truncation_psi=1.0),
                G_ema.mapping(z, [prompt]*len(z), truncation_psi=0.0),
            ]
            imgs = []
            for wi in ws:
                imgs.append(G_ema.synthesis(wi, noise_mode="const").cpu())
            prompts_results[prompt] = save_image_grid(torch.cat(imgs), drange=[-1, 1], grid_size=(n_samples, -1))
        save_prompts_dict(prompts_results, path=save_path)
    else:
        z = np.random.RandomState(0).randn(n_samples**2, G_ema.z_dim)
        z = torch.from_numpy(z).float().to(device)
        imgs = G_ema(z, c=None, noise_mode="const").cpu().numpy()
        save_image_grid(imgs, drange=[-1,1], grid_size=(n_samples, n_samples), fname=save_path)


def network_summaries(G: nn.Module, D: nn.Module, device: torch.device) -> None:
    z = torch.randn([1, G.z_dim], device=device)
    c = torch.randn([1, G.c_dim], device=device)
    img = misc.print_module_summary(G, [z, c])
    misc.print_module_summary(D, [img, c])


# Distributed
# ------------------------------------------------------------------------------------------

def sharded_all_mean(tensor: torch.Tensor, shard_size: int = 2**23) -> torch.Tensor:
    assert tensor.dim() == 1
    shards = tensor.tensor_split(math.ceil(tensor.numel() / shard_size))
    for shard in shards:
        torch.distributed.all_reduce(shard)
    tensor = torch.cat(shards) / dist.get_world_size()
    return tensor


def sync_grads(network: nn.Module, gain: Optional[int] = None) -> None:
    params = [param for param in network.parameters() if param.grad is not None]
    flat_grads = torch.cat([param.grad.flatten() for param in params])
    flat_grads = sharded_all_mean(flat_grads)
    flat_grads = flat_grads if gain is None else flat_grads * gain
    torch.nan_to_num(flat_grads, nan=0, posinf=1e5, neginf=-1e5, out=flat_grads)
    grads = flat_grads.split([param.numel() for param in params])
    for param, grad in zip(params, grads):
        param.grad = grad.reshape(param.size())


# Data
# ------------------------------------------------------------------------------------------

def split(arr: Union[list, np.ndarray, torch.Tensor], chunk_size: int, dim: int = 0) -> list:
    ''' equivalent to torch.Tensor.split, works for np/torch/list'''
    splits = int(np.ceil(len(arr) / chunk_size))
    return np.array_split(arr, splits, dim)


def fetch_data(
    training_set_iterator: Iterator,
    z_dim: int,
    device: torch.device,
    batches_num: int,
    batch_size: int,
    batch_gpu: int
) -> tuple[torch.Tensor, Union[list[str], None]]:
    # Get data and sample latents.
    real_img, real_cs = next(training_set_iterator)
    real_img = real_img.to(device).to(torch.float32) / 127.5 - 1
    gen_zs = torch.randn([batches_num * batch_size, z_dim], device = device)

    # Split for phases.
    real_img = split(real_img, batch_gpu)
    gen_zs = [split(gen_z, batch_gpu) for gen_z in split(gen_zs, batch_size)]
    real_cs = split(real_cs, batch_gpu)

    return real_img, real_cs, gen_zs


# Training
# ------------------------------------------------------------------------------------------

def partial_freeze(phase: dnnlib.EasyDict) -> None:
    if phase.name == 'D':
        phase.module.dino.requires_grad_(False)

    elif phase.name == 'G':
        trainable_layers = phase.module.trainable_layers

        phase.module.requires_grad_(False)
        for name, layer in phase.module.named_modules():
            should_train = any(layer_type in name for layer_type in trainable_layers)
            layer.requires_grad_(should_train)


def training_loop(
    run_dir                 = '.',      # Output directory.
    training_set_kwargs     = {},       # Options for training set.
    data_loader_kwargs      = {},       # Options for torch.utils.data.DataLoader.
    G_kwargs                = {},       # Options for generator network.
    D_kwargs                = {},       # Options for discriminator network.
    G_opt_kwargs            = {},       # Options for generator optimizer.
    D_opt_kwargs            = {},       # Options for discriminator optimizer.
    loss_kwargs             = {},       # Options for loss function.
    metrics                 = [],       # Metrics to evaluate during training.
    random_seed             = 0,        # Global random seed.
    batch_size              = 4,        # Total batch size for one training iteration. Can be larger than batch_gpu * num_gpus.
    batch_gpu               = 4,        # Number of samples processed at a time by one GPU.
    ema_kimg                = 10,       # Half-life of the exponential moving average (EMA) of generator weights.
    ema_rampup              = 0.05,     # EMA ramp-up coefficient. None = no rampup.
    total_kimg              = 25000,    # Total length of the training, measured in thousands of real images.
    kimg_per_tick           = 4,        # Progress snapshot interval.
    image_snapshot_ticks    = 50,       # How often to save image snapshots? None = disable.
    network_snapshot_ticks  = 50,       # How often to save network snapshots? None = disable.
    resume_pkl              = None,     # Network pickle to resume training from.
    resume_kimg             = 0,        # First kimg to report when resuming training.
    cudnn_benchmark         = True,     # Enable torch.backends.cudnn.benchmark?
    abort_fn                = None,     # Callback function for determining whether to abort training. Must return consistent results across ranks.
    progress_fn             = None,     # Callback function for updating training progress. Called for all ranks.
    device                  = torch.device('cuda'),
) -> None:

    # Initialize.
    start_time = time.time()
    np.random.seed(random_seed * dist.get_world_size() + dist.get_rank())
    torch.manual_seed(random_seed * dist.get_world_size() + dist.get_rank())
    torch.backends.cudnn.benchmark = cudnn_benchmark    # Improves training speed.
    torch.backends.cuda.matmul.allow_tf32 = False       # Improves numerical accuracy.
    torch.backends.cudnn.allow_tf32 = False             # Improves numerical accuracy.
    conv2d_gradfix.enabled = True                       # Improves training speed.

    # Select batch size per GPU.
    batch_gpu_total = batch_size // dist.get_world_size()
    if batch_gpu is None or batch_gpu > batch_gpu_total:
        batch_gpu = batch_gpu_total
    n_batch_acc = batch_gpu_total // batch_gpu
    assert batch_size == batch_gpu * n_batch_acc * dist.get_world_size()

    # Load training set. Choose between WDS and zip dataloader.
    dist.print0('Loading training set...')
    training_set = dnnlib.util.construct_class_by_name(**training_set_kwargs)
    if training_set_kwargs.class_name == 'training.data_wds.WdsWrapper':
        training_set_iterator = iter(wds_dataloader(training_set.urls, resolution=training_set.resolution, batch_size=batch_size//dist.get_world_size()))
    else:
        training_set_sampler = misc.InfiniteSampler(dataset=training_set, rank=dist.get_rank(), num_replicas=dist.get_world_size(), seed=random_seed)
        training_set_iterator = iter(torch.utils.data.DataLoader(dataset=training_set, sampler=training_set_sampler, batch_size=batch_size//dist.get_world_size(), **data_loader_kwargs))

    dist.print0('Num images: ', len(training_set))
    dist.print0('Image shape:', training_set.image_shape)
    dist.print0('Label shape:', training_set.label_shape)
    dist.print0()

    # Construct networks.
    dist.print0('Constructing networks...')
    G = dnnlib.util.construct_class_by_name(conditional=(training_set.label_dim>0), **G_kwargs).train().requires_grad_(False).to(device)
    G_ema = copy.deepcopy(G).eval()
    D = dnnlib.util.construct_class_by_name(c_dim=G.c_dim, **D_kwargs).train().requires_grad_(False).to(device)

    # Check for existing checkpoint
    data = {}
    if (resume_pkl is not None) and dist.get_rank() == 0:
        dist.print0(f'Resuming from "{resume_pkl}"')
        with dnnlib.util.open_url(resume_pkl) as f:
            data = dill.load(f)
        misc.copy_params_and_buffers(src_module=data['D'], dst_module=D, require_all=False)
        misc.copy_params_and_buffers(src_module=data['G_ema'], dst_module=G, require_all=False)
        misc.copy_params_and_buffers(src_module=data['G_ema'], dst_module=G_ema, require_all=False)
    del data

    # Print network summary tables.
    if dist.get_rank() == 0:
        network_summaries(G, D, device)

    # Distribute across GPUs.
    dist.print0(f'Distributing across {dist.get_world_size()} GPUs...')
    for module in [G, D, G_ema]:
        if module is not None and dist.get_world_size() > 1:
            for param in misc.params_and_buffers(module):
                torch.distributed.broadcast(param, src=0)

    # Setup training phases.
    dist.print0('Setting up training phases...')
    loss = dnnlib.util.construct_class_by_name(device=device, G=G, D=D, **loss_kwargs)
    phases = []

    for name, module, opt_kwargs in [('D', D, D_opt_kwargs), ('G', G, G_opt_kwargs)]:
        opt = dnnlib.util.construct_class_by_name(params=module.parameters(), **opt_kwargs)
        phases += [dnnlib.EasyDict(name=name, module=module, opt=opt, interval=1)]

    for phase in phases:
        phase.start_event = None
        phase.end_event = None
        if dist.get_rank() == 0:
            phase.start_event = torch.cuda.Event(enable_timing=True)
            phase.end_event = torch.cuda.Event(enable_timing=True)

    # Export sample images.
    dist.print0('Exporting sample images...')
    if dist.get_rank() == 0:
        grid_size, images, _ = setup_snapshot_image_grid(training_set)
        save_image_grid(images, drange=[0, 255], grid_size=grid_size, fname=os.path.join(run_dir, "reals.png"))

    # Initialize logs.
    dist.print0('Initializing logs...')
    stats_collector = training_stats.Collector(regex='.*')
    stats_metrics = dict()
    stats_jsonl = None
    stats_tfevents = None
    if dist.get_rank() == 0:
        stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'wt')
        stats_tfevents = tensorboard.SummaryWriter(run_dir)

    # Train.
    dist.print0(f'Training for {total_kimg} kimg...')
    dist.print0()

    cur_nimg = resume_kimg * 1000
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    batch_idx = 0
    if progress_fn is not None:
        progress_fn(cur_nimg // 1000, total_kimg)

    while True:
        # Get data
        phase_real_img, phase_real_c, all_gen_z = fetch_data(training_set_iterator, G.z_dim, device, len(phases), batch_size, batch_gpu)

        # Execute training phases.
        for phase, phase_gen_z in zip(phases, all_gen_z):
            if batch_idx % phase.interval != 0:
                continue
            if phase.start_event is not None:
                phase.start_event.record(torch.cuda.current_stream(device))

            # Enable/disable gradients.
            phase.module.requires_grad_(True)
            partial_freeze(phase)

            # Accumulate gradients.
            for real_img, real_c, gen_z in zip(phase_real_img, phase_real_c, phase_gen_z):
                loss.accumulate_gradients(phase=phase.name, real_img=real_img, c_raw=real_c, gen_z=gen_z, cur_nimg=cur_nimg)
            phase.module.requires_grad_(False)

            # Update weights.
            params = [param for param in phase.module.parameters() if param.grad is not None]
            if len(params) > 0:
                sync_grads(network=phase.module, gain=n_batch_acc)
            phase.opt.step()
            phase.opt.zero_grad(set_to_none=True)

            # Phase done.
            if phase.end_event is not None:
                phase.end_event.record(torch.cuda.current_stream(device))

        # Update G_ema.
        ema_nimg = ema_kimg * 1000
        if ema_rampup is not None:
            ema_nimg = min(ema_nimg, cur_nimg * ema_rampup)
        ema_beta = 0.5 ** (batch_size / max(ema_nimg, 1e-8))
        for p_ema, p in zip(G_ema.parameters(), G.parameters()):
            p_ema.copy_(p.lerp(p_ema, ema_beta))
        for b_ema, b in zip(G_ema.buffers(), G.buffers()):
            b_ema.copy_(b)

        # Update state.
        cur_nimg += batch_size
        batch_idx += 1

        # Perform maintenance tasks once per tick.
        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        # Print status line, accumulating the same information in training_stats.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<8.1f}"]
        fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
        fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
        fields += [f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"]
        torch.cuda.reset_peak_memory_stats()
        training_stats.report0('Timing/total_hours', (tick_end_time - start_time) / (60 * 60))
        training_stats.report0('Timing/total_days', (tick_end_time - start_time) / (24 * 60 * 60))
        dist.print0(' '.join(fields))

        # Check for abort.
        if (not done) and (abort_fn is not None) and abort_fn():
            done = True
            dist.print0()
            dist.print0('Aborting...')

        # Save image snapshot.
        if (dist.get_rank() == 0) and (image_snapshot_ticks is not None) and (done or cur_tick % image_snapshot_ticks == 0):
            save_samples(G_ema, run_dir=run_dir, device=device, suffix=f'{cur_nimg//1000:06d}')

        # Save network snapshot.
        data = None
        if (network_snapshot_ticks is not None) and (done or cur_tick % network_snapshot_ticks == 0):
            data = dict(G=G, D=D, G_ema=G_ema, training_set_kwargs=dict(training_set_kwargs))
            for key, value in data.items():
                if isinstance(value, torch.nn.Module):
                    value = copy.deepcopy(value).eval().requires_grad_(False)
                    misc.check_ddp_consistency(value, ignore_regex=r'.*\.[^.]+_(avg|ema)')
                    for param in misc.params_and_buffers(value):
                        torch.distributed.broadcast(param, src=0)
                    data[key] = value.cpu()
                del value  # conserve memory

            if dist.get_rank() == 0:
                snapshot_pkl = os.path.join(run_dir, f'network-snapshot-{cur_nimg//1000:06d}.pkl')
                with open(snapshot_pkl, 'wb') as f:
                    dill.dump(data, f)

        # Evaluate metrics.
        if cur_tick and (data is not None) and (len(metrics) > 0):
            dist.print0('Evaluating metrics...')
            for metric in metrics:
                result_dict = metric_main.calc_metric(metric=metric, G=data['G_ema'],
                    dataset_kwargs=training_set_kwargs, num_gpus=dist.get_world_size(), rank=dist.get_rank(), device=device)
                if dist.get_rank() == 0:
                    metric_main.report_metric(result_dict, run_dir=run_dir, snapshot_pkl=snapshot_pkl)
                stats_metrics.update(result_dict.results)
        del data # conserve memory

        # Collect statistics.
        for phase in phases:
            value = []
            if (phase.start_event is not None) and (phase.end_event is not None):
                phase.end_event.synchronize()
                value = phase.start_event.elapsed_time(phase.end_event)
            training_stats.report0('Timing/' + phase.name, value)
        stats_collector.update()
        stats_dict = stats_collector.as_dict()

        # Update logs.
        timestamp = time.time()
        if stats_jsonl is not None:
            fields = dict(stats_dict, timestamp=timestamp)
            stats_jsonl.write(json.dumps(fields) + '\n')
            stats_jsonl.flush()
        if stats_tfevents is not None:
            global_step = int(cur_nimg / 1e3)
            walltime = timestamp - start_time
            for name, value in stats_dict.items():
                stats_tfevents.add_scalar(name, value.mean, global_step=global_step, walltime=walltime)
            for name, value in stats_metrics.items():
                stats_tfevents.add_scalar(f'Metrics/{name}', value, global_step=global_step, walltime=walltime)
            stats_tfevents.flush()
        if progress_fn is not None:
            progress_fn(cur_nimg // 1000, total_kimg)

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    # Done.
    dist.print0()
    dist.print0('Exiting...')
