# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Miscellaneous utilities used internally by the quality metrics."""

import os
import random
import time
import hashlib
import copy
import uuid
from pathlib import Path
from typing import Iterator, Callable, Optional

import numpy as np
import torch
import dill
from tqdm import tqdm

import dnnlib

class MetricOptions:
    def __init__(
        self,
        G: Optional[torch.nn.Module] = None,
        G_kwargs: dict = {},
        dataset_kwargs: dict = {},
        num_gpus: int = 1,
        rank: int = 0,
        device: Optional[torch.device] = None,
        cache: bool =True,
        progress = None,
    ):
        assert 0 <= rank < num_gpus
        self.G              = G
        self.G_kwargs       = dnnlib.EasyDict(G_kwargs)
        self.dataset_kwargs = dnnlib.EasyDict(dataset_kwargs)
        self.num_gpus       = num_gpus
        self.rank           = rank
        self.device         = device if device is not None else torch.device('cuda')
        self.progress       = progress.sub() if progress is not None and rank == 0 else ProgressMonitor()
        self.cache          = cache

_feature_detector_cache = dict()

def get_feature_detector_name(url: str) -> str:
    return os.path.splitext(url.split('/')[-1])[0]

def get_feature_detector(
    url: str,
    device: torch.device = torch.device('cpu'),
    num_gpus: int = 1,
    rank: int = 0,
    verbose: bool = False,
) -> torch.nn.Module:
    assert 0 <= rank < num_gpus
    key = (url, device)
    if key not in _feature_detector_cache:
        is_leader = (rank == 0)
        if not is_leader and num_gpus > 1:
            torch.distributed.barrier() # leader goes first
        with dnnlib.util.open_url(url, verbose=(verbose and is_leader)) as f:
            _feature_detector_cache[key] = dill.load(f).to(device)
        if is_leader and num_gpus > 1:
            torch.distributed.barrier() # others follow
    return _feature_detector_cache[key]

def iterate_random_labels(opts: MetricOptions, batch_size: int) -> Iterator[torch.Tensor]:
    if opts.G.c_dim == 0:
        c = torch.zeros([batch_size, opts.G.c_dim], device=opts.device)
        while True:
            yield c
    else:
        dataset = dnnlib.util.construct_class_by_name(**opts.dataset_kwargs)
        while True:
            c = [dataset.get_label(np.random.randint(len(dataset))) for _i in range(batch_size)]
            if not dataset.labels_are_text:
                c = torch.from_numpy(np.stack(c)).pin_memory().to(opts.device)
            yield c

class FeatureStats:
    def __init__(self, capture_all: bool = False, capture_mean_cov: bool = False, max_items: bool = None):
        self.capture_all = capture_all
        self.capture_mean_cov = capture_mean_cov
        self.max_items = max_items
        self.num_items = 0
        self.num_features = None
        self.all_features = None
        self.raw_mean = None
        self.raw_cov = None

    def set_num_features(self, num_features: int):
        if self.num_features is not None:
            assert num_features == self.num_features
        else:
            self.num_features = num_features
            self.all_features = []
            self.raw_mean = np.zeros([num_features], dtype=np.float64)
            self.raw_cov = np.zeros([num_features, num_features], dtype=np.float64)

    def is_full(self) -> bool:
        return (self.max_items is not None) and (self.num_items >= self.max_items)

    def append(self, x) -> None:
        x = np.asarray(x, dtype=np.float32)
        assert x.ndim == 2
        if (self.max_items is not None) and (self.num_items + x.shape[0] > self.max_items):
            if self.num_items >= self.max_items:
                return
            x = x[:self.max_items - self.num_items]

        self.set_num_features(x.shape[1])
        self.num_items += x.shape[0]
        if self.capture_all:
            self.all_features.append(x)
        if self.capture_mean_cov:
            x64 = x.astype(np.float64)
            self.raw_mean += x64.sum(axis=0)
            self.raw_cov += x64.T @ x64

    def append_torch(self, x: torch.Tensor, num_gpus: int = 1, rank: int = 0) -> None:
        assert isinstance(x, torch.Tensor) and x.ndim == 2
        assert 0 <= rank < num_gpus
        if num_gpus > 1:
            ys = []
            for src in range(num_gpus):
                y = x.clone()
                torch.distributed.broadcast(y, src=src)
                ys.append(y)
            x = torch.stack(ys, dim=1).flatten(0, 1) # interleave samples
        self.append(x.cpu().numpy())

    def get_all(self) -> np.ndarray:
        assert self.capture_all
        return np.concatenate(self.all_features, axis=0)

    def get_all_torch(self) -> torch.Tensor:
        return torch.from_numpy(self.get_all())

    def get_mean_cov(self):
        assert self.capture_mean_cov
        mean = self.raw_mean / self.num_items
        cov = self.raw_cov / self.num_items
        cov = cov - np.outer(mean, mean)
        return mean, cov

    def save(self, pkl_file: str) -> None:
        with open(pkl_file, 'wb') as f:
            dill.dump(self.__dict__, f)

    @staticmethod
    def load(pkl_file: str):
        with open(pkl_file, 'rb') as f:
            s = dnnlib.EasyDict(dill.load(f))
        obj = FeatureStats(capture_all=s.capture_all, max_items=s.max_items)
        obj.__dict__.update(s)
        return obj

class ProgressMonitor:
    def __init__(self, tag: Optional[str] = None, num_items: Optional[int] = None, flush_interval: int = 1000, verbose: bool = False,
                 progress_fn: Optional[Callable] = None, pfn_lo: int = 0, pfn_hi: int = 1000, pfn_total: int = 1000):
        self.tag = tag
        self.num_items = num_items
        self.verbose = verbose
        self.flush_interval = flush_interval
        self.progress_fn = progress_fn
        self.pfn_lo = pfn_lo
        self.pfn_hi = pfn_hi
        self.pfn_total = pfn_total
        self.start_time = time.time()
        self.batch_time = self.start_time
        self.batch_items = 0
        if self.progress_fn is not None:
            self.progress_fn(self.pfn_lo, self.pfn_total)

    def update(self, cur_items: int) -> None:
        assert (self.num_items is None) or (cur_items <= self.num_items)
        if (cur_items < self.batch_items + self.flush_interval) and (self.num_items is None or cur_items < self.num_items):
            return
        cur_time = time.time()
        total_time = cur_time - self.start_time
        time_per_item = (cur_time - self.batch_time) / max(cur_items - self.batch_items, 1)
        if (self.verbose) and (self.tag is not None):
            print(f'{self.tag:<19s} items {cur_items:<7d} time {dnnlib.util.format_time(total_time):<12s} ms/item {time_per_item*1e3:.2f}')
        self.batch_time = cur_time
        self.batch_items = cur_items

        if (self.progress_fn is not None) and (self.num_items is not None):
            self.progress_fn(self.pfn_lo + (self.pfn_hi - self.pfn_lo) * (cur_items / self.num_items), self.pfn_total)

    def sub(self, tag: Optional[str] = None, num_items: Optional[int] = None, flush_interval: int = 1000, rel_lo: int = 0, rel_hi: int = 1):
        return ProgressMonitor(
            tag             = tag,
            num_items       = num_items,
            flush_interval  = flush_interval,
            verbose         = self.verbose,
            progress_fn     = self.progress_fn,
            pfn_lo          = self.pfn_lo + (self.pfn_hi - self.pfn_lo) * rel_lo,
            pfn_hi          = self.pfn_lo + (self.pfn_hi - self.pfn_lo) * rel_hi,
            pfn_total       = self.pfn_total,
        )

def compute_feature_stats_for_dataset(
    opts: MetricOptions,
    detector_url: str,
    detector_kwargs: dict = {},
    rel_lo: int = 0,
    rel_hi: int = 1,
    batch_size: int = 64,
    data_loader_kwargs: Optional[dict] = None,
    max_items: Optional[int] = None,
    shuffle_size: Optional[int] = None,
    **stats_kwargs,
) -> FeatureStats:

    if data_loader_kwargs is None:
        data_loader_kwargs = dict(pin_memory=True, num_workers=3, prefetch_factor=2)

    # Try to lookup from cache
    cache_file = None
    if opts.cache:
        det_name = get_feature_detector_name(detector_url)

        # Choose cache file name.
        dataset_name = Path(opts.dataset_kwargs.path).stem
        args = dict(dataset_kwargs=opts.dataset_kwargs, detector_url=detector_url, detector_kwargs=detector_kwargs, stats_kwargs=stats_kwargs, shuffle_size=shuffle_size)
        md5 = hashlib.md5(repr(sorted(args.items())).encode('utf-8'))
        cache_tag = f'{dataset_name}-{det_name}-{md5.hexdigest()}'
        cache_file = dnnlib.make_cache_dir_path('gan-metrics', cache_tag + '.pkl')

        # Check if the file exists (all processes must agree).
        flag = os.path.isfile(cache_file) if opts.rank == 0 else False
        if opts.num_gpus > 1:
            flag = torch.as_tensor(flag, dtype=torch.float32, device=opts.device)
            torch.distributed.broadcast(tensor=flag, src=0)
            flag = (float(flag.cpu()) != 0)

        # Load.
        if flag:
            return FeatureStats.load(cache_file)

    print('Calculating the stats for this dataset the first time\n')
    print(f'Saving them to {cache_file}')
    dataset = dnnlib.util.construct_class_by_name(**opts.dataset_kwargs)

    # Initialize.
    num_items = len(dataset)
    if max_items is not None:
        num_items = min(num_items, max_items)
    stats = FeatureStats(max_items=num_items, **stats_kwargs)
    progress = opts.progress.sub(tag='dataset features', num_items=num_items, rel_lo=rel_lo, rel_hi=rel_hi)

    # get detector
    detector = get_feature_detector(url=detector_url, device=opts.device, num_gpus=opts.num_gpus, rank=opts.rank, verbose=progress.verbose)

    # Main loop.
    item_subset = [(i * opts.num_gpus + opts.rank) % num_items for i in range((num_items - 1) // opts.num_gpus + 1)]
    if shuffle_size is not None:
        random.shuffle(item_subset)
        item_subset = item_subset[:shuffle_size]

    for images, _ in tqdm(torch.utils.data.DataLoader(dataset=dataset, sampler=item_subset, batch_size=batch_size, **data_loader_kwargs)):
        if images.shape[1] == 1:
            images = images.repeat([1, 3, 1, 1])

        with torch.no_grad():
            features = detector(images.to(opts.device), **detector_kwargs)
        stats.append_torch(features, num_gpus=opts.num_gpus, rank=opts.rank)
        progress.update(stats.num_items)

    # Save to cache.
    if cache_file is not None and opts.rank == 0:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        temp_file = cache_file + '.' + uuid.uuid4().hex
        stats.save(temp_file)
        os.replace(temp_file, cache_file)  # atomic
    return stats

def compute_feature_stats_for_generator(
    opts: MetricOptions,
    detector_url: str,
    detector_kwargs: dict,
    rel_lo: int = 0,
    rel_hi: int = 1,
    batch_size: int = 64,
    batch_gen: Optional[int] = None,
    **stats_kwargs,
) -> FeatureStats:

    if batch_gen is None:
        batch_gen = min(batch_size, 4)
    assert batch_size % batch_gen == 0

    # get truncation values
    truncation_psi = opts.G_kwargs.get('truncation_psi', 1.0)

    # Setup generator and labels.
    G = copy.deepcopy(opts.G).eval().requires_grad_(False).to(opts.device)
    c_iter = iterate_random_labels(opts=opts, batch_size=batch_gen)

    # Initialize.
    stats = FeatureStats(**stats_kwargs)
    assert stats.max_items is not None
    progress = opts.progress.sub(tag='generator features', num_items=stats.max_items, rel_lo=rel_lo, rel_hi=rel_hi)

    # get detector
    detector = get_feature_detector(url=detector_url, device=opts.device, num_gpus=opts.num_gpus, rank=opts.rank, verbose=progress.verbose)

    # Main loop.
    while not stats.is_full():
        images = []
        texts = []

        for _i in range(batch_size // batch_gen):
            G.mapping.num_broadcast = G.mapping.num_ws
            z = torch.randn([batch_gen, G.z_dim], device=opts.device)
            c = next(c_iter)

            # generate images
            w = G.mapping(z, c, truncation_psi=truncation_psi)
            img = G.synthesis(w)

            # keep track
            img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            images.append(img)
            texts.append(c)

        images = torch.cat(images)

        if 'texts' in detector_kwargs.keys():
            detector_kwargs['texts'] = np.concatenate(texts)

        if images.shape[1] == 1:
            images = images.repeat([1, 3, 1, 1])

        with torch.no_grad():
            features = detector(images.to(opts.device), **detector_kwargs)
        stats.append_torch(features, num_gpus=opts.num_gpus, rank=opts.rank)
        progress.update(stats.num_items)

    del detector  # free memory
    return stats
