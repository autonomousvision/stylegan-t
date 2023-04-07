# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""

import os
import re
from typing import List, Optional, Union
import PIL.Image

import numpy as np
import torch
import click
import dill
from tqdm import tqdm

import dnnlib


def parse_range(s: Union[str, List]) -> List[int]:
    '''Parse a comma separated list of numbers or ranges and return a list of ints.

    Example: '1,2,5-10' returns [1, 2, 5, 6, 7]
    '''
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges


def parse_vec2(s: Union[str, tuple[float, float]]) -> tuple[float, float]:
    '''Parse a floating point 2-vector of syntax 'a,b'.

    Example:
        '0,1' returns (0,1)
    '''
    if isinstance(s, tuple): return s
    parts = s.split(',')
    if len(parts) == 2:
        return (float(parts[0]), float(parts[1]))
    raise ValueError(f'cannot parse 2-vector {s}')


def make_transform(translate: tuple[float,float], angle: float) -> np.ndarray:
    m = np.eye(3)
    s = np.sin(angle/360.0*np.pi*2)
    c = np.cos(angle/360.0*np.pi*2)
    m[0][0] = c
    m[0][1] = s
    m[0][2] = translate[0]
    m[1][0] = -s
    m[1][1] = c
    m[1][2] = translate[1]
    return m


@click.command("cli", context_settings={'show_default': True})
@click.option('--network', 'network_pkl', help='Network pickle filename',                  type=str, required=True)
@click.option('--seeds',                  help='List of random seeds (e.g., \'0,1,4-6\')', type=parse_range, required=True)
@click.option('--prompt',                 help='Text prompt',                              type=str)
@click.option('--outdir',                 help='Where to save the output images',          type=str, required=True)
@click.option('--truncation',             help='Truncation strength',                      type=float, default=1.0)
@click.option('--noise-mode',             help='Noise mode',                               type=click.Choice(['const', 'random', 'none']), default='const')
@click.option('--translate',              help='Translate XY-coordinate (e.g. \'0.3,1\')', type=parse_vec2, default='0,0')
@click.option('--rotate',                 help='Rotation angle in degrees',                type=float, default=0)
@click.option('--device',                 help='CPU or GPU.',                              type=torch.device, default=torch.device('cuda'))
def generate_images(
    network_pkl: str,
    seeds: List[int],
    prompt: Optional[str],
    outdir: str,
    truncation: float,
    noise_mode: str,
    translate: tuple[float,float],
    rotate: float,
    device: torch.device,
) -> None:
    print(f'Loading networks from "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl) as f:
        G = dill.load(f)['G_ema']
    G = G.eval().requires_grad_(False).to(device)
    if G.c_dim > 1:
        assert prompt, "Provide a prompt for conditional generators."

    os.makedirs(outdir, exist_ok=True)

    for seed in tqdm(seeds):
        # Construct an inverse rotation/translation matrix and pass to the generator.  The
        # generator expects this matrix as an inverse to avoid potentially failing numerical
        # operations in the network.
        if hasattr(G.synthesis, 'input'):
            m = make_transform(translate, rotate)
            m = np.linalg.inv(m)
            G.synthesis.input.transform.copy_(torch.from_numpy(m))

        # Generate
        z = np.random.RandomState(seed).randn(1, G.z_dim)
        z = torch.from_numpy(z).float().to(device)
        w = G.mapping(z, [prompt]*len(z), truncation_psi=truncation)
        img = G.synthesis(w, noise_mode=noise_mode)

        # Save
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/seed{seed:04d}.png')


if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter
