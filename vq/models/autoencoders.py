__all__ = [
    'BaseEncoder',
    'BaseDecoder',
]

from abc import ABC, abstractmethod

import torch
from todd.runners import Memo
from torch import nn


class BaseEncoder(nn.Module, ABC):

    @property
    @abstractmethod
    def out_channels(self) -> int:
        pass

    @abstractmethod
    def forward(
        self,
        image: torch.Tensor,
        memo: Memo,
    ) -> tuple[torch.Tensor, Memo]:
        pass


class BaseDecoder(nn.Module, ABC):

    @property
    @abstractmethod
    def in_channels(self) -> int:
        pass

    @property
    @abstractmethod
    def last_parameter(self) -> nn.Parameter:
        pass

    @abstractmethod
    def forward(
        self,
        z: torch.Tensor,
        memo: Memo,
    ) -> tuple[torch.Tensor, Memo]:
        pass
