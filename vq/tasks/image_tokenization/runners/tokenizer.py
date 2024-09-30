__all__ = [
    'Tokenizer',
]

from typing import TypeVar, cast

import todd
from todd.bases.registries import Item
from todd.runners import Memo
from torch import nn

from vq.datasets import Batch
from vq.runners import BaseValidator

from ..models import BaseModel
from ..registries import VQITRunnerRegistry

T = TypeVar('T', bound=nn.Module)


@VQITRunnerRegistry.register_()
class Tokenizer(BaseValidator[T]):

    @classmethod
    def callbacks_build_pre_hook(
        cls,
        config: todd.Config,
        registry: todd.RegistryMeta,
        item: Item,
    ) -> todd.Config:
        config.update(tokenize=True)
        if not todd.Store.DRY_RUN:
            config.callbacks = [
                dict(
                    type='LogCallback',
                    interval=50,
                    collect_env=dict(),
                    with_file_handler=True,
                    eta=dict(type='EMA_ETA', ema=dict(decay=0.9)),
                ),
            ]
        return super().callbacks_build_pre_hook(config, registry, item)

    def _run_iter(self, batch: Batch, memo: Memo, *args, **kwargs) -> Memo:
        if todd.Store.DRY_RUN:
            return super()._run_iter(batch, memo, *args, **kwargs)
        model = cast(BaseModel, self.strategy.module)
        original_image = batch['original_image']
        image = batch['image']
        if todd.Store.cuda:
            original_image = original_image.cuda()
            image = image.cuda()
        memo.update(original_image=original_image, image=image)
        _, memo = model.encode_to_quant(image, memo)
        return memo
