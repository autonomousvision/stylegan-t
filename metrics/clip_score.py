# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

""" Calculate CLIP score """

import os
import dnnlib
from pathlib import Path

import dill

from metrics import metric_utils
from networks.clip import CLIP

def build_clip(url: str) -> None:
    clip = CLIP('ViT-g-14', pretrained='laion2b_s12b_b42k')
    Path(url).parent.mkdir(parents=True, exist_ok=True)
    with open(url, 'wb') as f:
        dill.dump(clip, f)

def compute_clip_score(opts, num_gen: int) -> tuple[float, float]:
    cache_dir = dnnlib.make_cache_dir_path('detectors')
    detector_url = os.path.join(cache_dir, 'clipvitg14.pkl')
    detector_kwargs = {'texts': None, 'div255': True}

    # If it does not exist, build and save CLIP.
    if not os.path.exists(detector_url) and opts.rank == 0:
        build_clip(detector_url)

    if opts.rank == 0:
        print(f"detector_url: {detector_url}")

    feats = metric_utils.compute_feature_stats_for_generator(
        opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
        capture_all=True, max_items=num_gen).get_all_torch()

    if opts.rank != 0:
        return float('nan'), float('nan')

    image_features, text_features = feats.tensor_split((feats.size(1)//2,), 1)
    score = (image_features * text_features).sum(-1).mean()
    return float(score)
