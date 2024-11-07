__all__ = [
    'ComposedConnector',
]

from typing import cast

import todd
import torch
from todd.bases.registries import BuildPreHookMixin, Item
from todd.patches.torch import Sequential
from todd.runners import Memo

from ..registries import VQITConnectorRegistry
from .base import BaseConnector


@VQITConnectorRegistry.register_()
class ComposedConnector(BuildPreHookMixin, BaseConnector):

    def __init__(self, *args, connectors: Sequential, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert cast(BaseConnector, connectors[0]).in_channels == \
            self._in_channels
        assert cast(BaseConnector, connectors[-1]).out_channels == \
            self._out_channels
        self._connectors = connectors

    @classmethod
    def connectors_build_pre_hook(
        cls,
        config: todd.Config,
        registry: todd.RegistryMeta,
        item: Item,
    ) -> todd.Config:
        in_channels = config.in_channels

        connector: BaseConnector
        connectors: list[BaseConnector] = []
        for connector in config.connectors:
            connector = VQITConnectorRegistry.build_or_return(
                connector,
                in_channels=in_channels,
                out_channels=config.out_channels,
            )
            in_channels = connector.out_channels
            connectors.append(connector)

        config.connectors = Sequential(*connectors, unpack_args=True)
        return config

    @classmethod
    def build_pre_hook(
        cls,
        config: todd.Config,
        registry: todd.RegistryMeta,
        item: Item,
    ) -> todd.Config:
        config = super().build_pre_hook(config, registry, item)
        config = cls.connectors_build_pre_hook(config, registry, item)
        return config

    def forward(
        self,
        x: torch.Tensor,
        memo: Memo,
    ) -> tuple[torch.Tensor, Memo]:
        return self._connectors(x, memo)
