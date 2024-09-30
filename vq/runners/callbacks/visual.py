__all__ = [
    'VisualMixin',
    'BatchedVisualCallback',
    'UnbatchedVisualCallback',
]

import pathlib
import re
from typing import TypeVar

import cv2
import einops
import torch
import torchvision
from todd.patches.torch import get_rank
from todd.runners import Memo
from todd.runners.callbacks import BaseCallback, IntervalMixin
from torch import nn

from ...datasets import Batch
from ..registries import VQCallbackRegistry

T = TypeVar('T', bound=nn.Module)


class VisualMixin(BaseCallback[T]):

    def __init__(self, *args, name: str, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._name = name

    @property
    def visual_dir(self) -> pathlib.Path:
        return self.runner.work_dir / self._name

    def visual(
        self,
        batch: Batch,
        memo: Memo,
        k: str,
        v: torch.Tensor,
    ) -> None:
        v = einops.rearrange(v, 'c h w -> h w c')
        image = v.cpu().numpy()
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        assert cv2.imwrite(str(self.visual_dir / k), image)

    def before_run(self, memo: Memo) -> None:
        super().before_run(memo)
        if get_rank() == 0:
            self.visual_dir.mkdir(parents=True, exist_ok=True)

    def before_run_iter(self, batch: Batch, memo: Memo) -> None:
        super().before_run_iter(batch, memo)
        memo[self._name] = dict()

    def after_run_iter(self, batch: Batch, memo: Memo) -> None:
        visual: Memo = memo.pop(self._name, dict())
        for k, v in visual.items():
            self.visual(batch, memo, k, v)
        super().after_run_iter(batch, memo)


@VQCallbackRegistry.register_()
class BatchedVisualCallback(VisualMixin[T], IntervalMixin[T], BaseCallback[T]):

    def __init__(
        self,
        *args,
        name: str = 'batched_visual',
        columns: int = 4,
        **kwargs,
    ) -> None:
        super().__init__(*args, name=name, **kwargs)
        self._columns = columns

    def _should_run_iter(self) -> bool:
        return super()._should_run_iter() and get_rank() == 0

    def visual(
        self,
        batch: Batch,
        memo: Memo,
        k: str,
        v: torch.Tensor,
    ) -> None:
        k = f'{self.runner.iter_}_{k}.png'
        v = torchvision.utils.make_grid(v, self._columns)
        return super().visual(batch, memo, k, v)

    def before_run_iter(self, batch: Batch, memo: Memo) -> None:
        super().before_run_iter(batch, memo)
        if not self._should_run_iter():
            memo.pop(self._name)


@VQCallbackRegistry.register_()
class UnbatchedVisualCallback(VisualMixin[T], BaseCallback[T]):

    def __init__(
        self,
        *args,
        name: str = 'unbatched_visual',
        regex: str,
        **kwargs,
    ) -> None:
        super().__init__(*args, name=name, **kwargs)
        self._regex = re.compile(regex)

    def visual(
        self,
        batch: Batch,
        memo: Memo,
        k: str,
        v: torch.Tensor,
    ) -> None:
        if not self._regex.fullmatch(k):
            return
        for id_, v_i in zip(batch['id_'], v):
            k_i = f'{k}_{id_.replace("/", "-")}.png'
            super().visual(batch, memo, k_i, v_i)
