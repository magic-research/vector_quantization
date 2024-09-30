__all__ = [
    'ImageLossMetric',
]

from typing import TypeVar

import einops
import torch
from todd.patches.py import get_
from todd.runners import Memo
from todd.runners.metrics import LossMetric
from torch import nn

from ...datasets import Batch
from ..base import BaseMixin
from ..registries import VQMetricRegistry

T = TypeVar('T', bound=nn.Module)


@VQMetricRegistry.register_()
class ImageLossMetric(LossMetric[T]):
    runner: BaseMixin[T]

    def __init__(self, *args, pred_image: str, image: str, **kwargs) -> None:
        inputs = dict(pred_image=pred_image, image=image)
        super().__init__(*args, inputs=inputs, **kwargs)

    def _forward(self, batch: Batch, memo: Memo) -> tuple[torch.Tensor, Memo]:
        inputs = {k: get_(memo, v) for k, v in self._inputs.items()}
        inputs = {
            k: self.runner.dataset.decode(v) / 255
            for k, v in inputs.items()
        }
        loss = self._loss(**inputs)
        loss = einops.reduce(loss, 'b ... -> b', reduction='mean')
        return loss, memo
