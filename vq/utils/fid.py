__all__ = [
    'FIDModel',
    'FIDCallback',
]

from typing import Any, TypeVar, cast

import todd.tasks.image_generation as ig
import torch
from todd.registries import ModelRegistry
from todd.runners import CallbackRegistry, Memo
from todd.runners.callbacks import BaseCallback
from todd.utils import Store
from torch import nn

from ..datasets.base import T
from ..runners import BaseValidator

ModuleType = TypeVar('ModuleType', bound=nn.Module)


@ModelRegistry.register_()
class FIDModel(ig.Statistician):

    def forward(  # type: ignore[override] # pylint: disable=arguments-differ
        self,
        runner: Any,
        batch: T,
        memo: Memo,
        *args,
        mode: None = None,
        **kwargs,
    ) -> Memo:
        assert mode is None
        original_image = batch['original_image']
        if Store.cuda:
            original_image = original_image.cuda()
        super().forward(original_image)
        return memo


@CallbackRegistry.register_()
class FIDCallback(BaseCallback[ModuleType]):
    runner: BaseValidator[ModuleType]

    def after_run(self, memo: Memo) -> None:
        super().after_run(memo)
        runner = self.runner
        model = cast(FIDModel, runner.strategy.module)
        statistics = model.summarize()
        torch.save(statistics, runner.dataset.fid_path)
        memo['statistics'] = statistics
