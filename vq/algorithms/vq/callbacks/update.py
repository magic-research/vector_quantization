__all__ = [
    'UpdateMixin',
]

import todd
import torch
from todd.bases.registries import BuildPreHookMixin, Item
from todd.utils import EMA

from vq.tasks.image_tokenization.models.quantizers.callbacks import (
    BaseCallback,
)


class UpdateMixin(BuildPreHookMixin, BaseCallback):

    def __init__(
        self,
        *args,
        ema: EMA | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if ema is not None:
            self._ema = ema

    @classmethod
    def ema_build_pre_hook_mixin(
        cls,
        config: todd.Config,
        registry: todd.RegistryMeta,
        item: Item,
    ) -> todd.Config:
        if (ema := config.get('ema')) is not None:
            config.ema = EMA(**ema)
        return config

    @classmethod
    def build_pre_hook(
        cls,
        config: todd.Config,
        registry: todd.RegistryMeta,
        item: Item,
    ) -> todd.Config:
        config = super().build_pre_hook(config, registry, item)
        config = cls.ema_build_pre_hook_mixin(config, registry, item)
        return config

    @property
    def with_ema(self) -> bool:
        return hasattr(self, '_ema')

    def _update_embedding(self, e: torch.Tensor) -> None:
        if todd.Store.DRY_RUN:
            assert todd.utils.is_sync(e)
        self.vector_quantizer.embedding.weight.data = e
