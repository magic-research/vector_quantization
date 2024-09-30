__all__ = [
    'BaseDiscriminator',
]

from abc import ABC, abstractmethod

import torch
from torch import nn


class BaseDiscriminator(nn.Module, ABC):

    @abstractmethod
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        pass
