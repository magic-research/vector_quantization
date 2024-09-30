__all__ = [
    'PatchGANDiscriminator',
]

import todd
import torch
from torch import nn

from ..registries import VQDiscriminatorRegistry
from .base import BaseDiscriminator

# TODO: refactor


@VQDiscriminatorRegistry.register_()
class PatchGANDiscriminator(BaseDiscriminator):

    def __init__(
        self,
        *args,
        in_channels: int = 3,
        width: int = 64,
        depth: int = 3,
        kernel_size: int = 4,
        padding: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        sequence: list[nn.Module] = [
            nn.Conv2d(
                in_channels,
                width,
                kernel_size=kernel_size,
                stride=2,
                padding=padding,
            ),
            nn.LeakyReLU(0.2, True),
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, depth):
            # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(
                    width * nf_mult_prev,
                    width * nf_mult,
                    kernel_size=kernel_size,
                    stride=2,
                    padding=padding,
                    bias=False,
                ),
                nn.BatchNorm2d(width * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**depth, 8)
        sequence += [
            nn.Conv2d(
                width * nf_mult_prev,
                width * nf_mult,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(width * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]

        sequence += [
            nn.Conv2d(
                width * nf_mult,
                1,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
            ),
        ]
        self._discriminator = nn.Sequential(*sequence)

    def init_weights(self, config: todd.Config) -> bool:

        def weights_init(m):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif classname.find('BatchNorm') != -1:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)

        # todd.logger.debug(f'Initializing {self.__class__.__name__} weights')
        self.apply(weights_init)
        return False

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self._discriminator(image)
