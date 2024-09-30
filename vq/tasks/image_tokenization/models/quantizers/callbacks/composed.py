__all__ = [
    'ComposedCallback',
]

from typing import Iterable, Literal, Mapping

import todd
import torch
from todd.bases.registries import BuildPreHookMixin, Item
from todd.runners import Memo
from todd.runners.utils import PriorityQueue

from ..registries import VQITQuantizerCallbackRegistry
from .base import BaseCallback

KT = Literal['bind', 'before_init_weights', 'after_init_weights',
             'before_encode', 'after_encode', 'before_decode', 'after_decode',
             'before_loss', 'after_loss']
Priority = Mapping[KT, int]


@VQITQuantizerCallbackRegistry.register_()
class ComposedCallback(BuildPreHookMixin, BaseCallback):

    def __init__(
        self,
        *args,
        priorities: Iterable[Priority],
        callbacks: Iterable[BaseCallback],
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._priority_queue = PriorityQueue(priorities, callbacks)

    @classmethod
    def build_pre_hook(
        cls,
        config: todd.Config,
        registry: todd.RegistryMeta,
        item: Item,
    ) -> todd.Config:
        config = super().build_pre_hook(config, registry, item)
        callbacks: Iterable[todd.Config] = config.callbacks
        config.priorities = [c.pop('priority', dict()) for c in callbacks]
        config.callbacks = [registry.build_or_return(c) for c in callbacks]
        return config

    def bind(self, *args, **kwargs) -> None:
        super().bind(*args, **kwargs)
        for callback in self._priority_queue('bind'):
            callback.bind(*args, **kwargs)

    def before_init_weights(self, *args, **kwargs) -> None:
        super().before_init_weights(*args, **kwargs)
        for callback in self._priority_queue('before_init_weights'):
            callback.before_init_weights(*args, **kwargs)

    def after_init_weights(self, config: todd.Config, recursive: bool) -> bool:
        recursive = super().after_init_weights(config, recursive)
        for callback in self._priority_queue('after_init_weights'):
            recursive = callback.after_init_weights(config, recursive)
        return recursive

    def before_encode(self, x: torch.Tensor, memo: Memo) -> torch.Tensor:
        x = super().before_encode(x, memo)
        for callback in self._priority_queue('before_encode'):
            x = callback.before_encode(x, memo)
        return x

    def after_encode(
        self,
        x: torch.Tensor,
        quant: torch.Tensor,
        memo: Memo,
    ) -> torch.Tensor:
        quant = super().after_encode(x, quant, memo)
        for callback in self._priority_queue('after_encode'):
            quant = callback.after_encode(x, quant, memo)
        return quant

    def before_decode(self, quant: torch.Tensor, memo: Memo) -> torch.Tensor:
        quant = super().before_decode(quant, memo)
        for callback in self._priority_queue('before_decode'):
            quant = callback.before_decode(quant, memo)
        return quant

    def after_decode(self, z: torch.Tensor, memo: Memo) -> torch.Tensor:
        z = super().after_decode(z, memo)
        for callback in self._priority_queue('after_decode'):
            z = callback.after_decode(z, memo)
        return z

    def before_loss(
        self,
        z: torch.Tensor,
        x: torch.Tensor,
        memo: Memo,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z, x = super().before_loss(z, x, memo)
        for callback in self._priority_queue('before_loss'):
            z, x = callback.before_loss(z, x, memo)
        return z, x

    def after_loss(self, loss: torch.Tensor, memo: Memo) -> torch.Tensor:
        loss = super().after_loss(loss, memo)
        for callback in self._priority_queue('after_loss'):
            loss = callback.after_loss(loss, memo)
        return loss
