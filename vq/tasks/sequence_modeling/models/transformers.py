__all__ = [
    'BaseTransformer',
]

import enum
from abc import ABC, abstractmethod
from typing import TypeVar

import todd
import todd.tasks.large_multimodal_model as lmm
import torch
from todd.bases.registries import BuildPreHookMixin, Item
from todd.runners import Memo
from torch import nn

from vq.utils import get_memo

from .registries import VQSMSamplerRegistry
from .samplers import BaseSampler

T = TypeVar('T', bound=enum.Enum)


class BaseTransformer(BuildPreHookMixin, nn.Module, ABC):

    def __init__(
        self,
        *args,
        vocabulary_size: int,
        sampler: BaseSampler,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._vocabulary_size = vocabulary_size
        self._sampler = sampler

    @classmethod
    def sampler_build_pre_hook(
        cls,
        config: todd.Config,
        registry: todd.RegistryMeta,
        item: Item,
    ) -> todd.Config:
        config.sampler = VQSMSamplerRegistry.build_or_return(config.sampler)
        return config

    @classmethod
    def build_pre_hook(
        cls,
        config: todd.Config,
        registry: todd.RegistryMeta,
        item: Item,
    ) -> todd.Config:
        config = super().build_pre_hook(config, registry, item)
        config = cls.sampler_build_pre_hook(config, registry, item)
        return config

    def sample(
        self,
        logits: torch.Tensor,
        codebook: lmm.Codebook[T],
        memo: Memo,
    ) -> tuple[torch.Tensor, Memo]:
        tokens, memo['sampler'] = self._sampler(
            logits,
            codebook.start,
            codebook.end,
            get_memo(memo, 'sampler'),
        )
        return tokens, memo

    @abstractmethod
    def _generate(
        self,
        tokens: torch.Tensor,
        length: int,
        codebook: lmm.Codebook[T],
        memo: Memo,
    ) -> tuple[torch.Tensor, Memo]:
        pass

    @torch.no_grad()
    def generate(
        self,
        tokens: torch.Tensor,
        length: int,
        codebook: lmm.Codebook[T],
        memo: Memo,
    ) -> tuple[torch.Tensor, Memo]:
        return self._generate(tokens, length, codebook, memo)

    @abstractmethod
    def forward(
        self,
        data: lmm.InterleavedData[T],
        memo: Memo,
    ) -> tuple[torch.Tensor, Memo]:
        pass
