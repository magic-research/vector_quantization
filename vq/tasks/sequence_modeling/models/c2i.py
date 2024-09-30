__all__ = [
    'C2I',
]

from typing import TypeVar

import todd
import todd.tasks.large_multimodal_model as lmm
import torch
from todd.bases.registries import Item
from todd.runners import Memo
from torch import nn

from vq.tasks.image_reconstruction import BaseModel as BaseIRModel

from ..runners import BaseMixin as BaseRunnerMixin
from .registries import VQSMModelRegistry
from .x2i import X2I

T = TypeVar('T', bound=nn.Module)


@VQSMModelRegistry.register_()
class C2I(X2I[lmm.C2IEnum]):

    @classmethod
    def vocabulary_size_build_pre_hook(
        cls,
        config: todd.Config,
        registry: todd.RegistryMeta,
        item: Item,
    ) -> todd.Config:
        if 'vocabulary_size' in config:
            return config
        config = cls.ir_build_pre_hook(
            config,
            registry,
            item,
        )
        cfg = bool(config.get('cfg'))
        ir: BaseIRModel = config.ir
        config.vocabulary_size = config.num_categories + cfg + ir.codebook_size
        return config

    @classmethod
    def build_pre_hook(
        cls,
        config: todd.Config,
        registry: todd.RegistryMeta,
        item: Item,
    ) -> todd.Config:
        config = cls.vocabulary_size_build_pre_hook(config, registry, item)
        return super().build_pre_hook(config, registry, item)

    @property
    def num_categories(self) -> int:
        return self._num_categories + bool(self._cfg)

    def uncondition_tokens(
        self,
        condition_tokens: torch.Tensor,
    ) -> torch.Tensor:
        return torch.full_like(condition_tokens, self._num_categories)

    def _data(
        self,
        runner: BaseRunnerMixin[T],
        memo: Memo,
    ) -> tuple[lmm.C2IData, Memo]:
        image_tokens, memo = self.encode_image_tokens(memo['image'], memo)
        memo['image_tokens'] = image_tokens

        category_tokens = memo['category']
        if self._cfg is not None:
            category_tokens, memo = self.dropout_tokens(category_tokens, memo)
        memo['category_tokens'] = category_tokens

        data = lmm.C2IData(
            category_tokens,
            image_tokens,
            self.num_categories,
            self._ir.codebook_size,
        )
        return data, memo
