__all__ = [
    'VQKDConnector',
]

import torch
from todd.runners import Memo
from torch import nn

from vq.tasks.image_tokenization.models import VQITConnectorRegistry
from vq.tasks.image_tokenization.models.connectors import BaseConnector


@VQITConnectorRegistry.register_()
class VQKDConnector(BaseConnector):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        conv1 = nn.Conv2d(
            self._in_channels,
            self._in_channels,
            3,
            padding=1,
        )
        tanh = nn.Tanh()
        conv2 = nn.Conv2d(
            self._in_channels,
            self._out_channels,
            3,
            padding=1,
        )
        self._sequential = nn.Sequential(conv1, tanh, conv2)

    def forward(
        self,
        x: torch.Tensor,
        memo: Memo,
    ) -> tuple[torch.Tensor, Memo]:
        x = self._sequential(x)
        return x, memo
