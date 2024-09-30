__all__ = [
    'FIDMetric',
]

from typing import TypeVar

import todd.tasks.image_generation as ig
from todd.patches.py import get_
from todd.patches.torch import load
from todd.runners import Memo
from todd.runners.metrics import BaseMetric
from torch import nn

from ...datasets import Batch
from ..base import BaseMixin
from ..registries import VQMetricRegistry

T = TypeVar('T', bound=nn.Module)


@VQMetricRegistry.register_()
class FIDMetric(BaseMetric[T]):
    runner: BaseMixin[T]

    def __init__(
        self,
        *args,
        pred: str,
        eps: float = 1e-6,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._pred = pred
        self._statistician = ig.Statistician()
        self._eps = eps

    def forward(self, batch: Batch, memo: Memo) -> Memo:
        pred_image = get_(memo, self._pred)
        image = self.runner.dataset.decode(pred_image)
        self._statistician(image)
        return memo

    def summary(self, memo: Memo) -> float:
        from ...utils import Store  # pylint: disable=import-outside-toplevel
        gt_statistics = load(
            self.runner.dataset.fid_path,
            'cpu',
            directory=Store.PRETRAINED,
        )
        pred_statistics = self._statistician.summarize()
        return ig.fid(gt_statistics, pred_statistics, self._eps)
