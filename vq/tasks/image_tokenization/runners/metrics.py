__all__ = [
    'CodebookMixin',
    'CodebookUsageMetric',
    'CodebookPPLMetric',
]

from abc import ABC, abstractmethod
from typing import TypeVar, cast

import torch
import torch.distributed as dist
from todd.patches.py_ import get_
from todd.runners import Memo
from todd.runners.metrics import BaseMetric
from torch import nn

from vq.datasets import Batch

from ..models import BaseModel
from .registries import VQITMetricRegistry

T = TypeVar('T', bound=nn.Module)


class CodebookMixin(BaseMetric[T], ABC):

    def __init__(self, *args, quant: str, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._quant = quant
        self._counts: torch.Tensor | int = 0

    def bind(self, *args, **kwargs) -> None:
        super().bind(*args, **kwargs)
        module = cast(BaseModel, self.runner.strategy.module)
        self._codebook_size = module.quantizer.codebook_size

    def forward(self, batch: Batch, memo: Memo) -> Memo:
        quant: torch.Tensor = get_(memo, self._quant)
        quant = quant.flatten()
        counts = torch.bincount(
            quant,
            minlength=self._codebook_size,
        )
        self._counts = self._counts + counts
        return memo

    @abstractmethod
    def _summary(self, memo: Memo, counts: torch.Tensor) -> float:
        pass

    def summary(self, memo: Memo) -> float:
        if isinstance(self._counts, int):
            return 0.
        counts = self._counts.clone()
        dist.all_reduce(counts)
        return self._summary(memo, counts)


@VQITMetricRegistry.register_()
class CodebookUsageMetric(CodebookMixin[T], BaseMetric[T]):

    def _summary(self, memo: Memo, counts: torch.Tensor) -> float:
        return counts.bool().sum().item() / self._codebook_size


@VQITMetricRegistry.register_()
class CodebookPPLMetric(CodebookMixin[T], BaseMetric[T]):

    def _summary(self, memo: Memo, counts: torch.Tensor) -> float:
        probabilities = counts / counts.sum()
        categorical = torch.distributions.Categorical(probabilities)
        entropy: torch.Tensor = categorical.entropy()
        return entropy.item()
