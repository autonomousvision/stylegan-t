# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Loss function."""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import RandomCrop

import dnnlib
from torch_utils import training_stats
from torch_utils.ops import upfirdn2d
from networks.clip import CLIP


class ProjectedGANLoss:
    def __init__(
        self,
        device: torch.device,
        G: nn.Module,
        D: nn.Module,
        blur_init_sigma: int = 2,
        blur_fade_kimg: int = 0,
        clip_weight: float = 0.0,
    ):
        super().__init__()
        self.device = device
        self.G = G
        self.D = D
        self.blur_init_sigma = blur_init_sigma
        self.blur_curr_sigma = blur_init_sigma
        self.blur_fade_kimg = blur_fade_kimg
        self.train_text_encoder = 'clip' in G.trainable_layers
        self.clip = CLIP().eval().to(self.device).requires_grad_(False)
        self.clip_weight = clip_weight

    @staticmethod
    def spherical_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = F.normalize(x, dim=-1)
        y = F.normalize(y, dim=-1)
        return (x * y).sum(-1).arccos().pow(2)

    @staticmethod
    def blur(img: torch.Tensor, blur_sigma: float) -> torch.Tensor:
        blur_size = np.floor(blur_sigma * 3)
        if blur_size > 0:
            with torch.autograd.profiler.record_function('blur'):
                f = torch.arange(-blur_size, blur_size + 1, device=img.device).div(blur_sigma).square().neg().exp2()
                img = upfirdn2d.filter2d(img, f / f.sum())
        return img

    def set_blur_sigma(self, cur_nimg: int) -> None:
        if self.blur_fade_kimg > 1:
            self.blur_curr_sigma = max(1 - cur_nimg / (self.blur_fade_kimg * 1e3), 0) * self.blur_init_sigma
        else:
            self.blur_curr_sigma = 0

    def run_G(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        ws = self.G.mapping(z, c)
        img = self.G.synthesis(ws)
        return img

    def run_D(self, img: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        if img.size(-1) > self.G.img_resolution:
            img = F.interpolate(img, self.G.img_resolution, mode='area')
        img = self.blur(img, self.blur_curr_sigma)
        return self.D(img, c)

    def accumulate_gradients(
        self,
        phase: dnnlib.EasyDict,
        real_img: torch.Tensor,
        c_raw: list[Optional[str]],
        gen_z: torch.Tensor,
        cur_nimg: int
    ) -> None:
        batch_size = real_img.size(0)
        self.set_blur_sigma(cur_nimg)

        # Encode text embedding for text conditional dataset
        c_enc = None
        if isinstance(c_raw[0], str):
            c_enc = self.clip.encode_text(c_raw)

        if phase == 'D':
            # Minimize logits for generated images.
            gen_img = self.run_G(gen_z, c=c_raw if self.train_text_encoder else c_enc)
            gen_logits = self.run_D(gen_img, c=c_enc)
            loss_gen = (F.relu(torch.ones_like(gen_logits) + gen_logits)).mean() / batch_size
            loss_gen.backward()

            # Maximize logits for real images.
            real_img_tmp = real_img.detach().requires_grad_(False)
            real_logits = self.run_D(real_img_tmp, c=c_enc)
            loss_real = (F.relu(torch.ones_like(real_logits) - real_logits)).mean() / batch_size
            loss_real.backward()

            # Collect stats.
            training_stats.report('Loss/scores/fake', gen_logits)
            training_stats.report('Loss/scores/real', real_logits)
            training_stats.report('Loss/signs/fake', gen_logits.sign())
            training_stats.report('Loss/signs/real', real_logits.sign())
            training_stats.report('Loss/D/loss', loss_gen + loss_real)

        elif phase == 'G':
            # Maximize logits for generated images.
            gen_img = self.run_G(gen_z, c=c_raw if self.train_text_encoder else c_enc)
            gen_logits = self.run_D(gen_img, c=c_enc)
            loss_gen = (-gen_logits).mean() / batch_size

            # Minimize spherical distance between image and text features
            clip_loss = 0
            if self.clip_weight > 0:
                if gen_img.size(-1) > 64:
                    gen_img = RandomCrop(64)(gen_img)
                gen_img = F.interpolate(gen_img, 224, mode='area')
                gen_img_features = self.clip.encode_image(gen_img.add(1).div(2))
                clip_loss = self.spherical_distance(gen_img_features, c_enc).mean()

            (loss_gen + self.clip_weight*clip_loss).backward()

            # Collect stats.
            training_stats.report('Loss/scores/fake', gen_logits)
            training_stats.report('Loss/signs/fake', gen_logits.sign())
            training_stats.report('Loss/G/loss', loss_gen)
            if self.clip_weight > 0:
                training_stats.report('Loss/G/clip_loss', clip_loss)
