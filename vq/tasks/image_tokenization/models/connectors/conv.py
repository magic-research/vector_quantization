__all__ = [
    'ConvConnector',
]

import todd
import torch
from todd.bases.registries import BuildPreHookMixin, Item
from todd.runners import Memo
from torch import nn

from ..registries import VQITConnectorRegistry
from .base import BaseConnector


@VQITConnectorRegistry.register_()
class ConvConnector(BuildPreHookMixin, BaseConnector):

    def __init__(self, *args, conv: nn.Conv2d, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert conv.in_channels == self._in_channels
        assert conv.out_channels == self._out_channels
        self._conv = conv

    @classmethod
    def conv_build_pre_hook(
        cls,
        config: todd.Config,
        registry: todd.RegistryMeta,
        item: Item,
    ) -> todd.Config:
        conv = config.conv if 'conv' in config else todd.Config(kernel_size=1)
        config.conv = nn.Conv2d(
            config.in_channels,
            config.out_channels,
            **conv,
        )
        return config

    @classmethod
    def build_pre_hook(
        cls,
        config: todd.Config,
        registry: todd.RegistryMeta,
        item: Item,
    ) -> todd.Config:
        config = super().build_pre_hook(config, registry, item)
        config = cls.conv_build_pre_hook(config, registry, item)
        return config

    def forward(
        self,
        x: torch.Tensor,
        memo: Memo,
    ) -> tuple[torch.Tensor, Memo]:
        x = self._conv(x)
        return x, memo
