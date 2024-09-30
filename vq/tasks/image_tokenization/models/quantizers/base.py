__all__ = [
    'BaseQuantizer',
]

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import todd
import torch
from todd.bases.registries import BuildPreHookMixin, Item
from todd.patches.torch import ModuleDict
from todd.runners import Memo
from torch import nn

from vq.utils import build_module_dict, get_memo

from .registries import (
    VQITQuantizerCallbackRegistry,
    VQITQuantizerLossRegistry,
)

if TYPE_CHECKING:
    from .callbacks import ComposedCallback


class BaseQuantizer(BuildPreHookMixin, nn.Module, ABC):

    def __init__(
        self,
        *args,
        callbacks: 'ComposedCallback',
        losses: ModuleDict,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._callbacks = callbacks
        self._losses = losses
        self._init()

    @classmethod
    def callbacks_build_pre_hook(
        cls,
        config: todd.Config,
        registry: todd.RegistryMeta,
        item: Item,
    ) -> todd.Config:
        from .callbacks import (  # pylint: disable=import-outside-toplevel
            ComposedCallback,
        )
        config.callbacks = VQITQuantizerCallbackRegistry.build(
            todd.Config(
                type=ComposedCallback.__name__,
                callbacks=config.get('callbacks', []),
            ),
        )
        return config

    @classmethod
    def losses_build_pre_hook(
        cls,
        config: todd.Config,
        registry: todd.RegistryMeta,
        item: Item,
    ) -> todd.Config:
        config.losses = build_module_dict(
            VQITQuantizerLossRegistry,
            config.get_config('losses'),
        )
        return config

    @classmethod
    def build_pre_hook(
        cls,
        config: todd.Config,
        registry: todd.RegistryMeta,
        item: Item,
    ) -> todd.Config:
        config = super().build_pre_hook(config, registry, item)
        config = cls.callbacks_build_pre_hook(config, registry, item)
        config = cls.losses_build_pre_hook(config, registry, item)
        return config

    def _init(self) -> None:
        self._callbacks.bind(self)

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        pass

    @property
    @abstractmethod
    def codebook_size(self) -> int:
        pass

    @property
    @abstractmethod
    def embeddings(self) -> torch.Tensor:
        pass

    def _init_weights(self, config: todd.Config) -> bool:
        return True

    def init_weights(self, config: todd.Config) -> bool:
        before_init_weights = config.pop('before_init_weights', todd.Config())
        after_init_weights = config.pop('after_init_weights', todd.Config())
        self._callbacks.before_init_weights(before_init_weights)
        recursive = self._init_weights(config)
        recursive = self._callbacks.after_init_weights(
            after_init_weights,
            recursive,
        )
        return recursive

    @abstractmethod
    def _encode(
        self,
        x: torch.Tensor,
        memo: Memo,
    ) -> tuple[torch.Tensor, Memo]:
        pass

    def encode(
        self,
        x: torch.Tensor,
        memo: Memo,
    ) -> tuple[torch.Tensor, torch.Tensor, Memo]:
        x = self._callbacks.before_encode(x, memo)
        quant, memo['encode'] = self._encode(x, get_memo(memo, 'encode'))
        quant = self._callbacks.after_encode(x, quant, memo)
        return x, quant, memo

    @abstractmethod
    def _decode(
        self,
        quant: torch.Tensor,
        memo: Memo,
    ) -> tuple[torch.Tensor, Memo]:
        pass

    def decode(
        self,
        quant: torch.Tensor,
        memo: Memo,
    ) -> tuple[torch.Tensor, Memo]:
        quant = self._callbacks.before_decode(quant, memo)
        z, memo['decode'] = self._decode(quant, get_memo(memo, 'decode'))
        z = self._callbacks.after_decode(z, memo)
        return z, memo

    def _loss(
        self,
        z: torch.Tensor,
        x: torch.Tensor,
        memo: Memo,
    ) -> tuple[torch.Tensor, Memo]:
        losses: dict[str, torch.Tensor] = self._losses(z, x, memo)
        memo.update(losses)
        loss = sum(losses.values(), x.new_zeros([]))
        return loss, memo

    def loss(
        self,
        z: torch.Tensor,
        x: torch.Tensor,
        memo: Memo,
    ) -> tuple[torch.Tensor, Memo]:
        z, x = self._callbacks.before_loss(z, x, memo)
        loss, memo['loss'] = self._loss(z, x, get_memo(memo, 'loss'))
        loss = self._callbacks.after_loss(loss, memo)
        return loss, memo

    def forward(
        self,
        x: torch.Tensor,
        memo: Memo,
    ) -> tuple[torch.Tensor, torch.Tensor, Memo]:
        x, quant, memo = self.encode(x, memo)
        memo.update(x=x, quant=quant)
        z, memo = self.decode(quant, memo)
        loss, memo = self.loss(z, x, memo)
        return z, loss, memo
