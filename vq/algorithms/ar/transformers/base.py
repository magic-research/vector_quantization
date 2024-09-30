__all__ = [
    'BaseTransformer',
]

import enum
from abc import abstractmethod
from typing import TypeVar

import todd.tasks.large_multimodal_model as lmm
import torch
from todd.runners import Memo

from vq.tasks.sequence_modeling.models import (
    BaseTransformer as BaseSMTransformer,
)

T = TypeVar('T', bound=enum.Enum)
KVCache = tuple[tuple[torch.Tensor, torch.Tensor], ...]


class BaseTransformer(BaseSMTransformer):

    @abstractmethod
    def _inference(
        self,
        tokens: torch.Tensor,
        kv_cache: KVCache | None,
        memo: Memo,
    ) -> tuple[torch.Tensor, KVCache, Memo]:
        pass

    @torch.no_grad()
    def inference(
        self,
        tokens: torch.Tensor,
        kv_cache: KVCache | None,
        memo: Memo,
    ) -> tuple[torch.Tensor, KVCache, Memo]:
        return self._inference(tokens, kv_cache, memo)

    def _generate(
        self,
        tokens: torch.Tensor,
        length: int,
        codebook: lmm.Codebook[T],
        memo: Memo,
    ) -> tuple[torch.Tensor, Memo]:
        assert tokens.shape[1] < length
        token = tokens
        kv_cache = None
        while tokens.shape[1] < length:
            logits, kv_cache, memo = self.inference(token, kv_cache, memo)
            token, memo = self.sample(logits[:, [-1]], codebook, memo)
            tokens = torch.cat((tokens, token), 1)
        assert tokens.shape[1] == length
        return tokens, memo
