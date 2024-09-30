__all__ = [
    'StyleGAN2Discriminator',
]

import math
from abc import ABC, abstractmethod

import einops
import todd
import torch
from mmcv.ops import FusedBiasLeakyReLU, upfirdn2d
from torch import nn

from ..registries import VQDiscriminatorRegistry
from .base import BaseDiscriminator


class EqualMixin(nn.Module, ABC):
    weight: torch.Tensor
    bias: torch.Tensor | nn.Parameter | None

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        weight = self.weight
        del self.weight

        self.weight = weight.data
        self._original_weight = weight

    def init_weights(self, config: todd.Config) -> bool:
        nn.init.normal_(self._original_weight, 0., 1.)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        return False

    @property
    @abstractmethod
    def _fan_in(self) -> int:
        pass

    def forward(self, *args, **kwargs) -> torch.Tensor:
        weight = self._original_weight
        self.weight = weight / self._fan_in**0.5
        return super().forward(*args, **kwargs)


class Linear(EqualMixin, nn.Linear):

    @property
    def _fan_in(self) -> int:
        return self.weight.shape[1]


class Conv2d(EqualMixin, nn.Conv2d):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *args,
        downsample: bool = False,
        **kwargs,
    ) -> None:
        if downsample:
            stride = 2
            padding = 0
        else:
            stride = 1
            padding = kernel_size // 2

        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            *args,
            **kwargs,
        )
        self._downsample = downsample

        if downsample:
            self._blur = Blur(kernel_size=kernel_size)

    @property
    def _fan_in(self) -> int:
        return math.prod(self.weight.shape[1:])

    def forward(  # pylint: disable=arguments-differ
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if self._downsample:
            x = self._blur(x)
        return super().forward(x)


class ActivateMixin(nn.Module, ABC):
    out_channels: int

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, bias=False, **kwargs)
        self._activate = FusedBiasLeakyReLU(self.out_channels)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        x = super().forward(*args, **kwargs)
        return self._activate(x)


class ActivateLinear(ActivateMixin, Linear):

    @property
    def out_channels(self) -> int:
        return self.out_features


class ActivateConv2d(ActivateMixin, Conv2d):
    pass


class Blur(nn.Module):

    def __init__(self, *args, kernel_size: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        kernel = torch.tensor([1., 3., 3., 1.])
        kernel = (
            einops.rearrange(kernel, 'k -> 1 k')
            * einops.rearrange(kernel, 'k -> k 1')
        )
        self.kernel = kernel / kernel.sum()

        pad = (kernel_size // 2 + 1, (kernel_size + 1) // 2)
        self._pad = pad

    @property
    def kernel(self) -> torch.Tensor:
        return self.get_buffer('_kernel')

    @kernel.setter
    def kernel(self, value: torch.Tensor) -> None:
        self.register_buffer('_kernel', value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return upfirdn2d(x, self.kernel, padding=self._pad)


class Residual(nn.Module):

    def __init__(
        self,
        *args,
        in_channels: int,
        out_channels: int,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._residual = nn.Sequential(
            ActivateConv2d(in_channels, in_channels, 3),
            ActivateConv2d(in_channels, out_channels, 3, downsample=True),
        )
        self._shortcut = Conv2d(
            in_channels,
            out_channels,
            1,
            downsample=True,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._shortcut(x) + self._residual(x)
        return x / 2**0.5


class Std(nn.Module):

    def __init__(
        self,
        *args,
        batch_groups: int = 4,
        eps: float = 1e-8,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._batch_groups = batch_groups
        self._eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape

        bg = min(b, self._batch_groups)
        y: torch.Tensor = einops.rearrange(
            x,
            '(bg b) c h w -> bg b c h w',
            bg=bg,
        )
        y = y.var(0, False) + self._eps
        y = y.sqrt()
        y = einops.reduce(y, 'b c h w -> b', 'mean')
        y = einops.repeat(y, 'b -> (bg b) 1 h w', bg=bg, h=h, w=w)

        x = torch.cat([x, y], 1)
        return x


@VQDiscriminatorRegistry.register_()
class StyleGAN2Discriminator(BaseDiscriminator):
    CHANNELS = {
        4: 512,
        8: 512,
        16: 512,
        32: 512,
        64: 512,
        128: 256,
        256: 128,
        512: 64,
        1024: 32,
    }

    def __init__(self, *args, image_size: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        channels = [
            self.CHANNELS[2**i]
            for i in range(int(math.log2(image_size)), 1, -1)
        ]

        self._layers = nn.Sequential(
            ActivateConv2d(3, channels[0], 1),
            *[
                Residual(in_channels=ic, out_channels=oc)
                for ic, oc in zip(channels[:-1], channels[1:])
            ],
            Std(),
            ActivateConv2d(channels[-1] + 1, self.CHANNELS[4], 3),
        )

        self._projector = nn.Sequential(
            ActivateLinear(self.CHANNELS[4] * 4 * 4, self.CHANNELS[4]),
            Linear(self.CHANNELS[4], 1),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x: torch.Tensor = self._layers(image)
        x = x.flatten(1)
        x = self._projector(x)
        return x
