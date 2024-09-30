__all__ = [
    'BaseMixin',
    'BaseTrainer',
    'BaseValidator',
]

from abc import ABC
from typing import Any, TypeVar

import todd
import torch
from todd.bases.registries import Item
from todd.runners import Memo
from todd.runners.callbacks import AutocastCallback
from torch import nn

from ..datasets import BaseMixin as BaseDatasetMixin
from ..datasets import Batch
from ..registries import VQRunnerRegistry
from .callbacks import UnbatchedVisualCallback
from .registries import VQCallbackRegistry

T = TypeVar('T', bound=nn.Module)


class BaseMixin(todd.runners.BaseRunner[T], ABC):
    _dataset: BaseDatasetMixin
    dataset: BaseDatasetMixin

    @classmethod
    def callbacks_build_pre_hook(
        cls,
        config: todd.Config,
        registry: todd.RegistryMeta,
        item: Item,
    ) -> todd.Config:
        if config.pop('autocast'):
            assert todd.Store.cuda
            dtype = (
                torch.bfloat16
                if torch.cuda.is_bf16_supported() else torch.float16
            )
            autocast_callback = dict(
                type=AutocastCallback.__name__,
                autocast=todd.Config(device_type='cuda', dtype=dtype),
            )
            config.callbacks = list(config.callbacks) + [autocast_callback]
        return super().callbacks_build_pre_hook(config, registry, item)


@VQRunnerRegistry.register_()
class BaseTrainer(BaseMixin[T], todd.runners.IterBasedTrainer[T]):

    def _run_iter(
        self,
        batch: Batch,
        memo: Memo,
        *args,
        mode: Any = 'train',
        **kwargs,
    ) -> Memo:
        return super()._run_iter(batch, memo, *args, mode=mode, **kwargs)


@VQRunnerRegistry.register_()
class BaseValidator(BaseMixin[T], todd.runners.Validator[T]):

    @classmethod
    def callbacks_build_pre_hook(
        cls,
        config: todd.Config,
        registry: todd.RegistryMeta,
        item: Item,
    ) -> todd.Config:
        callbacks = list(config.callbacks)
        if (visual := config.pop('visual', None)) is not None:
            unbatched_visual_callback = todd.Config(
                type=(
                    VQCallbackRegistry.__name__ + '.'
                    + UnbatchedVisualCallback.__name__
                ),
                regex=visual,
            )
            callbacks.append(unbatched_visual_callback)
        if config.pop('tokenize', False):
            tokenize_callback = todd.Config(
                type=(
                    'VQCallbackRegistry.VQITCallbackRegistry.TokenizeCallback'
                ),
            )
            callbacks.append(tokenize_callback)
        config.callbacks = callbacks
        return super().callbacks_build_pre_hook(config, registry, item)

    def _run_iter(
        self,
        batch: Batch,
        memo: Memo,
        *args,
        mode: Any = None,
        **kwargs,
    ) -> Memo:
        return super()._run_iter(batch, memo, *args, mode=mode, **kwargs)
