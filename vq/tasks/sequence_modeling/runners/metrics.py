__all__ = [
    'AccuracyMetric',
]

from typing import TypeVar

import torch
from todd.patches.py_ import get_
from todd.runners import Memo
from todd.runners.metrics import Metric
from torch import nn

from vq.datasets import Batch

from .registries import VQSMMetricRegistry

T = TypeVar('T', bound=nn.Module)


@VQSMMetricRegistry.register_()
class AccuracyMetric(Metric[T]):

    def __init__(
        self,
        *args,
        pred: str,
        target: str,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._pred = pred
        self._target = target

    def _forward(self, batch: Batch, memo: Memo) -> tuple[torch.Tensor, Memo]:
        log: Memo | None = memo.get('log')
        pred: torch.Tensor = get_(memo, self._pred)
        target: torch.Tensor = get_(memo, self._target)
        assert not pred.dtype.is_floating_point
        assert not target.dtype.is_floating_point
        pred = pred.reshape(pred.shape[0], -1)
        target = target.reshape(target.shape[0], -1)
        accuracy = pred == target
        accuracy = accuracy.float().mean(-1)
        if log is not None:
            log[self._name] = f'{accuracy.mean():.3f}'
        return accuracy, memo
