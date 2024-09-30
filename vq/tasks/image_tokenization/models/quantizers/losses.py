__all__ = [
    'BaseLoss',
]

from abc import ABC, abstractmethod

import torch
import torch.distributed
from todd.models import losses
from todd.runners import Memo


class BaseLoss(losses.BaseLoss, ABC):

    @abstractmethod
    def forward(  # pylint: disable=arguments-differ
        self,
        z: torch.Tensor,
        x: torch.Tensor,
        memo: Memo,
    ) -> torch.Tensor:
        pass
