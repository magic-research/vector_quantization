__all__ = [
    'BaseConnector',
]

import torch
from todd.runners import Memo
from torch import nn

from ..registries import VQITConnectorRegistry


@VQITConnectorRegistry.register_()
class BaseConnector(nn.Module):

    def __init__(
        self,
        *args,
        in_channels: int,
        out_channels: int,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._in_channels = in_channels
        self._out_channels = out_channels

    @property
    def in_channels(self) -> int:
        return self._in_channels

    @property
    def out_channels(self) -> int:
        return self._out_channels

    def forward(
        self,
        x: torch.Tensor,
        memo: Memo,
    ) -> tuple[torch.Tensor, Memo]:
        assert x.shape[1] == self._in_channels == self._out_channels
        return x, memo
