__all__ = [
    'HFTransformer',
]

import enum
from abc import abstractmethod
from typing import TypeVar

import todd
import todd.tasks.large_multimodal_model as lmm
import torch
from todd.bases.registries import Item
from todd.runners import Memo
from transformers import PreTrainedModel

from .base import BaseTransformer, KVCache

T = TypeVar('T', bound=enum.Enum)


class HFTransformer(BaseTransformer):

    def __init__(self, *args, transformer: PreTrainedModel, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        transformer.resize_token_embeddings(self._vocabulary_size)
        self._transformer = transformer

    @classmethod
    @abstractmethod
    def transformer_build_pre_hook(
        cls,
        config: todd.Config,
        registry: todd.RegistryMeta,
        item: Item,
    ) -> todd.Config:
        pass

    @classmethod
    def build_pre_hook(
        cls,
        config: todd.Config,
        registry: todd.RegistryMeta,
        item: Item,
    ) -> todd.Config:
        config = super().build_pre_hook(config, registry, item)
        config = cls.transformer_build_pre_hook(config, registry, item)
        return config

    def init_weights(self, config: todd.Config) -> bool:
        return False

    def _inference(
        self,
        tokens: torch.Tensor,
        kv_cache: KVCache | None,
        memo: Memo,
    ) -> tuple[torch.Tensor, KVCache, Memo]:
        output = self._transformer(tokens, past_key_values=kv_cache)
        return output['logits'], output['past_key_values'], memo

    def forward(
        self,
        data: lmm.InterleavedData[T],
        memo: Memo,
    ) -> tuple[torch.Tensor, Memo]:
        tokens = data.tokens
        output = self._transformer(tokens, labels=tokens)
        memo['logits'] = output['logits']
        return output['loss'], memo
