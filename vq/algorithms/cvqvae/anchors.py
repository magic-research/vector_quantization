__all__ = [
    'BaseAnchor',
    'NearestAnchor',
    'MultinomialAnchor',
    'CachedAnchor',
]

import random
from abc import ABC, abstractmethod
from typing import Mapping

import einops
import todd
import torch
import torch.distributed
from todd.patches.torch import all_gather, get_world_size
from todd.runners import Memo
from torch import nn

from .registries import AnchorRegistry


class BaseAnchor(nn.Module, ABC):

    def __init__(self, *args, sync: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._sync = sync

    @abstractmethod
    def _anchors(
        self,
        x: torch.Tensor,
        e: torch.Tensor,
        d: torch.Tensor,
        quant: torch.Tensor,
        p: torch.Tensor,
        memo: Memo,
    ) -> tuple[torch.Tensor, Memo]:
        pass

    def forward(
        self,
        x: torch.Tensor,
        e: torch.Tensor,
        d: torch.Tensor,
        quant: torch.Tensor,
        p: torch.Tensor,
        memo: Memo | None = None,
    ) -> tuple[torch.Tensor, Memo]:
        if self._sync:
            x = torch.cat(all_gather(x))
            if todd.Store.DRY_RUN:
                assert todd.utils.is_sync(e)
            d = torch.cat(all_gather(d))
            quant = torch.cat(all_gather(quant))
            p = torch.stack(all_gather(p))
            p = einops.reduce(p, 'ws p -> p', 'mean')
        if memo is None:
            memo = todd.Config()
        anchors, memo = self._anchors(x, e, d, quant, p, memo)
        if self._sync:
            if todd.Store.DRY_RUN:
                assert todd.utils.is_sync(anchors)
        else:
            if get_world_size() > 1:
                torch.distributed.all_reduce(anchors)
                anchors /= get_world_size()
        return anchors, memo


@AnchorRegistry.register_()
class NearestAnchor(BaseAnchor):

    def _anchors(
        self,
        x: torch.Tensor,
        e: torch.Tensor,
        d: torch.Tensor,
        quant: torch.Tensor,
        p: torch.Tensor,
        memo: Memo,
    ) -> tuple[torch.Tensor, Memo]:
        indices = d.argmin(0)
        anchors = x[indices]
        return anchors, memo


@AnchorRegistry.register_()
class MultinomialAnchor(BaseAnchor):

    def _anchors(
        self,
        x: torch.Tensor,
        e: torch.Tensor,
        d: torch.Tensor,
        quant: torch.Tensor,
        p: torch.Tensor,
        memo: Memo,
    ) -> tuple[torch.Tensor, Memo]:
        d_transposed = einops.rearrange(d, 'x e -> e x')
        indices = d_transposed.softmax(1).multinomial(1)
        indices = einops.rearrange(indices, 'e 1 -> e')
        anchors = x[indices]
        return anchors, memo


@AnchorRegistry.register_()
class CachedAnchor(BaseAnchor):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._update_cache(torch.empty(0))

    @property
    def cache(self) -> torch.Tensor:
        return self.get_buffer('_cache')

    def _update_cache(self, value: torch.Tensor) -> None:
        self.register_buffer('_cache', value)

    def _load_from_state_dict(
        self,
        state_dict: Mapping[str, torch.Tensor],
        prefix: str,
        *args,
        **kwargs,
    ) -> None:
        cache = state_dict.get(f'{prefix}_cache')
        if cache is not None:
            self.cache.resize_(cache.shape)
        return super()._load_from_state_dict(
            state_dict,
            prefix,
            *args,
            **kwargs,
        )

    def _anchors(
        self,
        x: torch.Tensor,
        e: torch.Tensor,
        d: torch.Tensor,
        quant: torch.Tensor,
        p: torch.Tensor,
        memo: Memo,
    ) -> tuple[torch.Tensor, Memo]:
        if x.shape[0] < d.shape[1] and self.cache.numel() > 0:
            x = torch.cat([x, self.cache])
        indices = (
            torch.randperm(d.shape[1]) if x.shape[0] <= d.shape[1] else
            random.sample(range(x.shape[0]), d.shape[1])
        )
        if x.shape[0] < d.shape[1]:
            missing = torch.rand(
                d.shape[1] - x.shape[0],
                x.shape[1],
                device=x.device,
            )
            x = torch.cat([x, missing])
        anchors = x[indices]  # type: ignore[index]
        return anchors, memo

    def forward(self, *args, **kwargs) -> tuple[torch.Tensor, Memo]:
        anchors, memo = super().forward(*args, **kwargs)
        self._update_cache(anchors.detach())
        return anchors, memo
