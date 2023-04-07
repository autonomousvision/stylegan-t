# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
Generator architecture from
"StyleGAN-T: Unlocking the Power of GANs for Fast Large-Scale Text-to-Image Synthesis".
"""

from typing import Union, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from torch_utils import misc
from torch_utils.ops import upfirdn2d, conv2d_resample, bias_act, fma
from networks.shared import FullyConnectedLayer, MLP
from networks.clip import CLIP


def is_list_of_strings(arr: Any) -> bool:
    if arr is None: return False
    is_list = isinstance(arr, list) or isinstance(arr, np.ndarray) or  isinstance(arr, tuple)
    entry_is_str = isinstance(arr[0], str)
    return is_list and entry_is_str


def normalize_2nd_moment(x: torch.Tensor, dim: int = 1, eps: float = 1e-8) -> torch.Tensor:
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()


def modulated_conv2d(
    x: torch.Tensor,                                 # Input tensor of shape [batch_size, in_channels, in_height, in_width].
    weight: torch.Tensor,                            # Weight tensor of shape [out_channels, in_channels, kernel_height, kernel_width].
    styles: torch.Tensor,                            # Modulation coefficients of shape [batch_size, in_channels].
    noise: Optional[torch.Tensor] = None,            # Optional noise tensor to add to the output activations.
    up: int = 1,                                     # Integer upsampling factor.
    down: int = 1,                                   # Integer downsampling factor.
    padding: int = 0,                                # Padding with respect to the upsampled image.
    resample_filter: Optional[list[int]] = None,     # Low-pass filter to apply when resampling activations.
    demodulate: bool = True,                         # Apply weight demodulation?
    flip_weight: bool = True,                        # False = convolution, True = correlation (matches torch.nn.functional.conv2d).
    fused_modconv: bool = True,                      # Perform modulation, convolution, and demodulation as a single fused operation?
) -> torch.Tensor:
    batch_size = x.shape[0]
    out_channels, in_channels, kh, kw = weight.shape
    misc.assert_shape(weight, [out_channels, in_channels, kh, kw]) # [OIkk]
    misc.assert_shape(x, [batch_size, in_channels, None, None]) # [NIHW]
    misc.assert_shape(styles, [batch_size, in_channels]) # [NI]

    # Pre-normalize inputs to avoid FP16 overflow.
    if x.dtype == torch.float16 and demodulate:
        weight = weight * (1 / np.sqrt(in_channels * kh * kw) / weight.norm(float('inf'), dim=[1,2,3], keepdim=True)) # max_Ikk
        styles = styles / styles.norm(float('inf'), dim=1, keepdim=True) # max_I

    # Calculate per-sample weights and demodulation coefficients.
    w = None
    dcoefs = None
    if demodulate or fused_modconv:
        w = weight.unsqueeze(0) # [NOIkk]
        w = w * styles.reshape(batch_size, 1, -1, 1, 1) # [NOIkk]
    if demodulate:
        dcoefs = (w.square().sum(dim=[2,3,4]) + 1e-8).rsqrt() # [NO]
    if demodulate and fused_modconv:
        w = w * dcoefs.reshape(batch_size, -1, 1, 1, 1) # [NOIkk]

    # Execute by scaling the activations before and after the convolution.
    if not fused_modconv:
        x = x * styles.to(x.dtype).reshape(batch_size, -1, 1, 1)
        x = conv2d_resample.conv2d_resample(x=x, w=weight.to(x.dtype), f=resample_filter, up=up, down=down, padding=padding, flip_weight=flip_weight)
        if demodulate and noise is not None:
            x = fma.fma(x, dcoefs.to(x.dtype).reshape(batch_size, -1, 1, 1), noise.to(x.dtype))
        elif demodulate:
            x = x * dcoefs.to(x.dtype).reshape(batch_size, -1, 1, 1)
        elif noise is not None:
            x = x.add_(noise.to(x.dtype))
        return x

    # Execute as one fused op using grouped convolution.
    with misc.suppress_tracer_warnings(): # this value will be treated as a constant
        batch_size = int(batch_size)
    misc.assert_shape(x, [batch_size, in_channels, None, None])
    x = x.reshape(1, -1, *x.shape[2:])
    w = w.reshape(-1, in_channels, kh, kw)
    x = conv2d_resample.conv2d_resample(x=x, w=w.to(x.dtype), f=resample_filter, up=up, down=down, padding=padding, groups=batch_size, flip_weight=flip_weight)
    x = x.reshape(batch_size, -1, *x.shape[2:])
    if noise is not None:
        x = x.add_(noise)
    return x


class GroupNorm32(nn.GroupNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).type(x.dtype)


class StyleSplit(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs):
        super().__init__()
        self.proj = FullyConnectedLayer(in_channels, 3*out_channels, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        m1, m2, m3 = x.chunk(3, 1)
        return m1 * m2 + m3


class SynthesisInput(torch.nn.Module):
    def __init__(
        self,
        w_dim: int,          # Intermediate latent (W) dimensionality.
        channels: int,       # Number of output channels.
        size: int,           # Output spatial size.
        sampling_rate: int,  # Output sampling rate.
        bandwidth: int,      # Output bandwidth.
    ):
        super().__init__()
        self.w_dim = w_dim
        self.channels = channels
        self.size = np.broadcast_to(np.asarray(size), [2])
        self.sampling_rate = sampling_rate
        self.bandwidth = bandwidth

        # Draw random frequencies from uniform 2D disc.
        freqs = torch.randn([self.channels, 2])
        radii = freqs.square().sum(dim=1, keepdim=True).sqrt()
        freqs /= radii * radii.square().exp().pow(0.25)
        freqs *= bandwidth
        phases = torch.rand([self.channels]) - 0.5

        # Setup parameters and buffers.
        self.weight = Parameter(torch.randn([self.channels, self.channels]))
        self.affine = FullyConnectedLayer(w_dim, 4, weight_init=0, bias_init=[1,0,0,0])

        self.register_buffer('transform', torch.eye(3, 3)) # User-specified inverse transform wrt. resulting image.
        self.register_buffer('freqs', freqs)
        self.register_buffer('phases', phases)

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        # Introduce batch dimension.
        transforms = self.transform.unsqueeze(0) # [batch, row, col]
        freqs = self.freqs.unsqueeze(0) # [batch, channel, xy]
        phases = self.phases.unsqueeze(0) # [batch, channel]

        # Apply learned transformation.
        t = self.affine(w) # t = (r_c, r_s, t_x, t_y)
        t = t / t[:, :2].norm(dim=1, keepdim=True) # t' = (r'_c, r'_s, t'_x, t'_y)
        m_r = torch.eye(3, device=w.device).unsqueeze(0).repeat([w.shape[0], 1, 1]) # Inverse rotation wrt. resulting image.
        m_r[:, 0, 0] = t[:, 0]  # r'_c
        m_r[:, 0, 1] = -t[:, 1] # r'_s
        m_r[:, 1, 0] = t[:, 1]  # r'_s
        m_r[:, 1, 1] = t[:, 0]  # r'_c
        m_t = torch.eye(3, device=w.device).unsqueeze(0).repeat([w.shape[0], 1, 1]) # Inverse translation wrt. resulting image.
        m_t[:, 0, 2] = -t[:, 2] # t'_x
        m_t[:, 1, 2] = -t[:, 3] # t'_y
        transforms = m_r @ m_t @ transforms # First rotate resulting image, then translate, and finally apply user-specified transform.

        # Transform frequencies.
        phases = phases + (freqs @ transforms[:, :2, 2:]).squeeze(2)
        freqs = freqs @ transforms[:, :2, :2]

        # Dampen out-of-band frequencies that may occur due to the user-specified transform.
        amplitudes = (1 - (freqs.norm(dim=2) - self.bandwidth) / (self.sampling_rate / 2 - self.bandwidth)).clamp(0, 1)

        # Construct sampling grid.
        theta = torch.eye(2, 3, device=w.device)
        theta[0, 0] = 0.5 * self.size[0] / self.sampling_rate
        theta[1, 1] = 0.5 * self.size[1] / self.sampling_rate
        grids = F.affine_grid(theta.unsqueeze(0), [1, 1, self.size[1], self.size[0]], align_corners=False)

        # Compute Fourier features.
        x = (grids.unsqueeze(3) @ freqs.permute(0, 2, 1).unsqueeze(1).unsqueeze(2)).squeeze(3) # [batch, height, width, channel]
        x = x + phases.unsqueeze(1).unsqueeze(2)
        x = torch.sin(x * (np.pi * 2))
        x = x * amplitudes.unsqueeze(1).unsqueeze(2)

        # Apply trainable mapping.
        weight = self.weight / np.sqrt(self.channels)
        x = x @ weight.t()

        # Ensure correct shape.
        x = x.permute(0, 3, 1, 2) # [batch, channel, height, width]
        misc.assert_shape(x, [w.shape[0], self.channels, int(self.size[1]), int(self.size[0])])
        return x.contiguous()

    def extra_repr(self) -> str:
        return '\n'.join([
            f'w_dim={self.w_dim:d}, channels={self.channels:d}, size={list(self.size)},',
            f'sampling_rate={self.sampling_rate:g}, bandwidth={self.bandwidth:g}'])


class SynthesisLayer(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,                        # Number of input channels.
        out_channels: int,                       # Number of output channels.
        w_dim: int,                              # Intermediate latent (W) dimensionality.
        resolution: int,                         # Resolution of this layer.
        kernel_size: int = 3,                    # Convolution kernel size.
        up: int = 1,                             # Integer upsampling factor.
        use_noise: bool = True,                  # Enable noise input?
        activation: str = 'lrelu',               # Activation function: 'relu', 'lrelu', etc.
        resample_filter: list[int] = [1,3,3,1],  # Low-pass filter to apply when resampling activations.
        conv_clamp: Optional[int] = None,        # Clamp the output of convolution layers to +-X, None = disable clamping.
        channels_last: bool = False,             # Use channels_last format for the weights?
        layer_scale_init: float = 1e-5,          # Initial value of layer scale.
        residual: bool = False,                  # Residual convolution?
        gn_groups: int = 32,                     # Number of groups for GroupNorm
    ):
        super().__init__()
        if residual: assert in_channels == out_channels

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.w_dim = w_dim
        self.resolution = resolution
        self.up = up
        self.use_noise = use_noise
        self.activation = activation
        self.conv_clamp = conv_clamp
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.padding = kernel_size // 2
        self.act_gain = bias_act.activation_funcs[activation].def_gain

        self.residual = residual

        if use_noise:
            self.register_buffer('noise_const', torch.randn([resolution, resolution]))
            self.noise_strength = Parameter(torch.zeros([]))

        self.affine = StyleSplit(w_dim, in_channels, bias_init=1)

        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        self.weight = Parameter(torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format))
        self.bias = Parameter(torch.zeros([out_channels]))

        if self.residual:
            assert up == 1
            self.norm = GroupNorm32(gn_groups, out_channels)
            self.gamma = Parameter(layer_scale_init * torch.ones([1, out_channels, 1, 1])).to(memory_format=memory_format)

    def forward(
        self,
        x: torch.Tensor,
        w: torch.Tensor,
        noise_mode: str = 'random',
        fused_modconv: bool = True,
        gain: int = 1,
    ) -> torch.Tensor:
        dtype = x.dtype
        in_resolution = self.resolution // self.up
        misc.assert_shape(x, [None, self.in_channels, in_resolution, in_resolution])

        noise = None
        if self.use_noise and noise_mode == 'random':
            noise = torch.randn([x.shape[0], 1, self.resolution, self.resolution], device=x.device) * self.noise_strength
        if self.use_noise and noise_mode == 'const':
            noise = self.noise_const * self.noise_strength

        flip_weight = (self.up == 1)  # slightly faster
        styles = self.affine(w)

        if self.residual:
            x = self.norm(x)

        y = modulated_conv2d(x=x, weight=self.weight, styles=styles, noise=noise, up=self.up, fused_modconv=fused_modconv,
                             padding=self.padding, resample_filter=self.resample_filter, flip_weight=flip_weight)
        y = y.to(dtype)

        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        y = bias_act.bias_act(y, self.bias.to(x.dtype), act=self.activation, gain=act_gain, clamp=act_clamp)

        if self.residual:
            y = self.gamma * y
            y = y.to(dtype).add_(x).mul(np.sqrt(2))

        return y

    def extra_repr(self) -> str:
        return ' '.join([
            f'in_channels={self.in_channels:d}, out_channels={self.out_channels:d}, w_dim={self.w_dim:d},',
            f'resolution={self.resolution:d}, up={self.up}, activation={self.activation:s}'])


class ToRGBLayer(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        w_dim: int,
        kernel_size: int = 1,
        conv_clamp: Optional[int] = None,
        channels_last: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.w_dim = w_dim
        self.conv_clamp = conv_clamp
        self.affine = StyleSplit(w_dim, in_channels, bias_init=1)

        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        self.weight = Parameter(0.1*torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format))
        self.bias = Parameter(torch.zeros([out_channels]))
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))

    def forward(self, x: torch.Tensor, w: torch.Tensor, fused_modconv: bool=True) -> torch.Tensor:
        styles = self.affine(w) * self.weight_gain
        x = modulated_conv2d(x=x, weight=self.weight, styles=styles, demodulate=False)
        x = bias_act.bias_act(x, self.bias.to(x.dtype), clamp=self.conv_clamp)
        return x

    def extra_repr(self) -> str:
        return f'in_channels={self.in_channels:d}, out_channels={self.out_channels:d}, w_dim={self.w_dim:d}'


class SynthesisBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,                               # Number of input channels, 0 = first block.
        out_channels: int,                              # Number of output channels.
        w_dim: int,                                     # Intermediate latent (W) dimensionality.
        resolution: int,                                # Resolution of this block.
        img_channels: int,                              # Number of output color channels.
        is_last: bool,                                  # Is this the last block?
        num_res_blocks: int = 1,                            # Number of conv layers per block.
        architecture: str = 'orig',                     # Architecture: 'orig', 'skip'.
        resample_filter: list[int] = [1,3,3,1],         # Low-pass filter to apply when resampling activations.
        conv_clamp: int = 256,                          # Clamp the output of convolution layers to +-X, None = disable clamping.
        use_fp16: bool = False,                         # Use FP16 for this block?
        fp16_channels_last: bool = False,               # Use channels-last memory format with FP16?
        fused_modconv_default: Any = 'inference_only',  # Default value of fused_modconv.
        **layer_kwargs,                                 # Arguments for SynthesisLayer.
    ):
        assert architecture in ['orig', 'skip']
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.w_dim = w_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.is_last = is_last
        self.architecture = architecture
        self.use_fp16 = use_fp16
        self.channels_last = (use_fp16 and fp16_channels_last)
        self.fused_modconv_default = fused_modconv_default
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.num_conv = 0
        self.num_torgb = 0

        if in_channels == 0:
            self.input = SynthesisInput(w_dim=self.w_dim, channels=out_channels, size=resolution, sampling_rate=resolution, bandwidth=2)
            self.num_conv += 1

        if in_channels != 0:
            self.conv0 = SynthesisLayer(in_channels, out_channels, w_dim=w_dim, resolution=resolution, up=2,
                resample_filter=resample_filter, conv_clamp=conv_clamp, channels_last=self.channels_last, **layer_kwargs)
            self.num_conv += 1

        convs = []
        for _ in range(num_res_blocks):
            convs.append(SynthesisLayer(out_channels, out_channels, w_dim=w_dim, resolution=resolution,
                                        conv_clamp=conv_clamp, channels_last=self.channels_last, **layer_kwargs))
            convs.append(SynthesisLayer(out_channels, out_channels, w_dim=w_dim, resolution=resolution,
                                        conv_clamp=conv_clamp, channels_last=self.channels_last,
                                        residual=True, **layer_kwargs))

        self.convs1 = torch.nn.ModuleList(convs)
        self.num_conv += len(convs)

        if is_last or architecture == 'skip':
            self.torgb = ToRGBLayer(out_channels, img_channels, w_dim=w_dim,
                conv_clamp=conv_clamp, channels_last=self.channels_last)
            self.num_torgb += 1

    def forward(
        self,
        x: torch.Tensor,
        img: Optional[torch.Tensor],
        ws: torch.Tensor,
        force_fp32: bool = False,
        fused_modconv: bool = True,
        **layer_kwargs,
    ) -> Union[torch.Tensor, Union[torch.Tensor, None]]:
        w_iter = iter(ws.unbind(dim=1))
        if ws.device.type != 'cuda':
            force_fp32 = True
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
        memory_format = torch.channels_last if self.channels_last and not force_fp32 else torch.contiguous_format
        if fused_modconv is None:
            fused_modconv = self.fused_modconv_default
        if fused_modconv == 'inference_only':
            fused_modconv = (not self.training)

        # Input.
        if self.in_channels == 0:
            x = self.input(next(w_iter))
        else:
            misc.assert_shape(x, [None, self.in_channels, self.resolution // 2, self.resolution // 2])
        x = x.to(dtype=dtype, memory_format=memory_format)

        # Main layers.
        if self.in_channels == 0:
            for conv in self.convs1:
                x = conv(x, next(w_iter), fused_modconv=fused_modconv, gain=np.sqrt(0.5), **layer_kwargs)
        else:
            x = self.conv0(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
            for conv in self.convs1:
                x = conv(x, next(w_iter), fused_modconv=fused_modconv, gain=np.sqrt(0.5), **layer_kwargs)

        # ToRGB.
        if img is not None:
            misc.assert_shape(img, [None, self.img_channels, self.resolution // 2, self.resolution // 2])
            img = upfirdn2d.upsample2d(img, self.resample_filter)

        if self.is_last or self.architecture == 'skip':
            y = self.torgb(x, next(w_iter), fused_modconv=fused_modconv)
            y = y.to(dtype=torch.float32, memory_format=torch.contiguous_format)
            img = img.add_(y) if img is not None else y

        assert x.dtype == dtype
        assert img is None or img.dtype == torch.float32
        return x, img

    def extra_repr(self) -> str:
        return f'resolution={self.resolution:d}, architecture={self.architecture:s}'


class SynthesisNetwork(torch.nn.Module):
    def __init__(
        self,
        w_dim: int,                 # Intermediate latent (W) dimensionality.
        img_resolution: int,        # Output image resolution.
        img_channels: int = 3,      # Number of color channels.
        channel_base: int = 32768,  # Overall multiplier for the number of channels.
        channel_max: int = 512,     # Maximum number of channels in any layer.
        num_fp16_res: int = 4,      # Use FP16 for the N highest resolutions.
        base_mult: int = 3,         # Start resolution (SG2: 2, SG3: 4, SG-T: 3).
        num_res_blocks: int = 3,        # Number of residual blocks.
        **block_kwargs,             # Arguments for SynthesisBlock.
    ):
        assert img_resolution >= 4 and img_resolution & (img_resolution - 1) == 0
        super().__init__()
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.num_fp16_res = num_fp16_res
        self.block_resolutions = [2 ** i for i in range(base_mult, self.img_resolution_log2 + 1)]
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        self.num_ws = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res // 2] if res > self.block_resolutions[0] else 0
            out_channels = channels_dict[res]
            use_fp16 = (res >= fp16_resolution)
            is_last = (res == self.img_resolution)
            block = SynthesisBlock(in_channels, out_channels, w_dim=w_dim, resolution=res, img_channels=img_channels, is_last=is_last, use_fp16=use_fp16, num_res_blocks=num_res_blocks, **block_kwargs)
            self.num_ws += block.num_conv
            if is_last:
                self.num_ws += block.num_torgb
            setattr(self, f'b{res}', block)

    def forward(self, ws: torch.Tensor, **block_kwargs) -> torch.Tensor:
        block_ws = []
        with torch.autograd.profiler.record_function('split_ws'):
            ws = ws.to(torch.float32)
            w_idx = 0
            for res in self.block_resolutions:
                block = getattr(self, f"b{res}")
                block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
                w_idx += block.num_conv

        x = img = None
        for res, cur_ws in zip(self.block_resolutions, block_ws):
            block = getattr(self, f'b{res}')
            x, img = block(x, img, cur_ws, **block_kwargs)
        return img

    def extra_repr(self) -> str:
        return ' '.join([
            f'w_dim={self.w_dim:d}, num_ws={self.num_ws:d},',
            f'img_resolution={self.img_resolution:d}, img_channels={self.img_channels:d},',
            f'num_fp16_res={self.num_fp16_res:d}'])


class MappingNetwork(torch.nn.Module):
    def __init__(
        self,
        z_dim: int,                   # Input latent (Z) dimensionality, 0 = no latent.
        conditional: bool = True,     # Text conditional?
        num_layers: int = 2,          # Number of mapping layers.
        activation: str = 'lrelu',    # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier: float = 0.01,  # Learning rate multiplier for the mapping layers.
        x_avg_beta: float = 0.995,    # Decay for tracking the moving average of W during training.
    ):
        super().__init__()
        self.z_dim = z_dim
        self.x_avg_beta = x_avg_beta
        self.num_ws = None

        self.mlp = MLP([z_dim]*(num_layers+1), activation=activation,
                       lr_multiplier=lr_multiplier, linear_out=True)

        if conditional:
            self.clip = CLIP()
            del self.clip.model.visual # only using the text encoder
            self.c_dim = self.clip.txt_dim
        else:
            self.c_dim = 0

        self.w_dim = self.c_dim + self.z_dim
        self.register_buffer('x_avg', torch.zeros([self.z_dim]))

    def forward(
        self,
        z: torch.Tensor,
        c: Union[None, torch.Tensor, list[str]],
        truncation_psi: float = 1.0,
    ) -> torch.Tensor:
        misc.assert_shape(z, [None, self.z_dim])

        # Forward pass.
        x = self.mlp(normalize_2nd_moment(z))

        # Update moving average.
        if self.x_avg_beta is not None and self.training:
            self.x_avg.copy_(x.detach().mean(0).lerp(self.x_avg, self.x_avg_beta))

        # Apply truncation.
        if truncation_psi != 1:
            assert self.x_avg_beta is not None
            x = self.x_avg.lerp(x, truncation_psi)

        # Build latent.
        if self.c_dim > 0:
            assert c is not None
            c = self.clip.encode_text(c) if is_list_of_strings(c) else c
            w = torch.cat([x, c], 1)
        else:
            w = x

        # Broadcast latent codes.
        if self.num_ws is not None:
            w = w.unsqueeze(1).repeat([1, self.num_ws, 1])

        return w


class Generator(torch.nn.Module):
    def __init__(
        self,
        z_dim: int,                  # Input latent (Z) dimensionality, 0 = no latent.
        conditional: bool,           # Text conditional?
        img_resolution: int,         # Output image resolution.
        img_channels: int = 3,       # Number of output color channels.
        train_mode: str = 'all',     # Control which layers are trainable.
        synthesis_kwargs: dict = {},
    ):
        super().__init__()
        self.z_dim = z_dim
        self.conditional = conditional
        self.img_resolution = img_resolution
        self.img_channels = img_channels

        self.mapping = MappingNetwork(z_dim=z_dim, conditional=conditional)
        self.synthesis = SynthesisNetwork(w_dim=self.mapping.w_dim, img_resolution=img_resolution,
                                          img_channels=img_channels, **synthesis_kwargs)

        self.w_dim = self.synthesis.w_dim
        self.c_dim = self.mapping.c_dim
        self.num_ws = self.synthesis.num_ws
        self.mapping.num_ws = self.num_ws

        # Set trainable layers.
        self.train_mode = train_mode
        if train_mode == 'all':
            self.trainable_layers = ['synthesis', 'mapping.mlp']
        elif train_mode == 'text_encoder':
            self.trainable_layers = ['clip']
        elif train_mode == 'freeze64':
            self.trainable_layers = [f"b{x}" for x in self.synthesis.block_resolutions if x > 64]
            self.trainable_layers += ['torgb']

    def forward(
        self,
        z: torch.Tensor,
        c: Union[None, torch.Tensor, list[str]],
        truncation_psi: float = 1.0, **synthesis_kwargs
    ) -> torch.Tensor:
        ws = self.mapping(z, c, truncation_psi=truncation_psi)
        img = self.synthesis(ws, **synthesis_kwargs)
        return img
