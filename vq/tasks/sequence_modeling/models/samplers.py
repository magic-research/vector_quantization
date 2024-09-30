__all__ = [
    'BaseSampler',
    'TopKTopPSampler',
    'CFGSampler',
]

from abc import ABC

import einops
import todd
import torch
from todd.bases.registries import BuildPreHookMixin, Item
from todd.runners import Memo
from torch import nn
from transformers import top_k_top_p_filtering

from .registries import VQSMSamplerRegistry


@VQSMSamplerRegistry.register_()
class BaseSampler(nn.Module, ABC):

    def sample(
        self,
        logits: torch.Tensor,
        memo: Memo,
    ) -> tuple[torch.Tensor, Memo]:
        tokens = logits.softmax(-1).multinomial(1)
        return tokens, memo

    @torch.no_grad()
    def forward(
        self,
        logits: torch.Tensor,
        start: int,
        end: int,
        memo: Memo,
    ) -> tuple[torch.Tensor, Memo]:
        shape = logits.shape
        logits = logits.reshape(-1, shape[-1])
        logits = logits[:, start:end]
        tokens, memo = self.sample(logits, memo)
        tokens = tokens + start
        tokens = tokens.reshape(shape[:-1])
        return tokens, memo


@VQSMSamplerRegistry.register_()
class TopKTopPSampler(BaseSampler):

    def __init__(
        self,
        *args,
        temperature: float = 1.0,
        top_k: int = 600,
        top_p: float = 0.92,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._temperature = temperature
        self._top_k = top_k
        self._top_p = top_p

    def sample(
        self,
        logits: torch.Tensor,
        memo: Memo,
    ) -> tuple[torch.Tensor, Memo]:
        logits = logits / self._temperature
        logits = top_k_top_p_filtering(logits, self._top_k, self._top_p)
        return super().sample(logits, memo)


@VQSMSamplerRegistry.register_()
class CFGSampler(BuildPreHookMixin, BaseSampler):

    def __init__(
        self,
        *args,
        sampler: BaseSampler,
        alpha: float,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._sampler = sampler
        self._alpha = alpha

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
        memo: Memo,
    ) -> tuple[torch.Tensor, Memo]:
        assert logits.shape[0] % 2 == 0
        unconditional_logits, conditional_logits = logits.chunk(2)
        cfg_logits = ((1 - self._alpha) * unconditional_logits
                      + self._alpha * conditional_logits)
        tokens, memo = self._sampler.sample(cfg_logits, memo)
        tokens = einops.repeat(tokens, 'b ... -> (two b) ...', two=2)
        return tokens, memo
