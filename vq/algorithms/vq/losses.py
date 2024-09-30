__all__ = [
    'CodebookLoss',
    'CommitmentLoss',
    'VQGANLoss',
]

from abc import ABC

import todd
import torch
import torch.distributed
from todd.bases.registries import BuildPreHookMixin
from todd.bases.registries.base import Item
from todd.models import losses
from todd.runners import Memo

from vq.tasks.image_tokenization.models.quantizers import (
    BaseLoss,
    VQITQuantizerLossRegistry,
)


class MSELoss(BaseLoss, BuildPreHookMixin, ABC):

    def __init__(self, *args, mse: losses.MSELoss, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._mse = mse

    @classmethod
    def build_pre_hook(
        cls,
        config: todd.Config,
        registry: todd.RegistryMeta,
        item: Item,
    ) -> todd.Config:
        config = super().build_pre_hook(config, registry, item)
        config.mse = losses.MSELoss(**config.get_config('mse'))
        return config


@VQITQuantizerLossRegistry.register_()
class CodebookLoss(MSELoss):

    def forward(
        self,
        z: torch.Tensor,
        x: torch.Tensor,
        memo: Memo,
    ) -> torch.Tensor:
        return self._mse(z, x.detach())


@VQITQuantizerLossRegistry.register_()
class CommitmentLoss(MSELoss):

    def forward(
        self,
        z: torch.Tensor,
        x: torch.Tensor,
        memo: Memo | None = None,
    ) -> torch.Tensor:
        return self._mse(z.detach(), x)


@VQITQuantizerLossRegistry.register_()
class VQGANLoss(BaseLoss):

    def __init__(
        self,
        *args,
        codebook: CodebookLoss,
        commitment: CommitmentLoss,
        beta: float = 0.25,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._codebook = codebook
        self._commitment = commitment
        self._beta = beta

    @classmethod
    def codebook_build_pre_hook(
        cls,
        config: todd.Config,
        registry: todd.RegistryMeta,
        item: Item,
    ) -> todd.Config:
        config.codebook = VQITQuantizerLossRegistry.build(
            config.get_config('codebook'),
            type=CodebookLoss.__name__,
        )
        return config

    @classmethod
    def commitment_build_pre_hook(
        cls,
        config: todd.Config,
        registry: todd.RegistryMeta,
        item: Item,
    ) -> todd.Config:
        config.commitment = VQITQuantizerLossRegistry.build(
            config.get_config('commitment'),
            type=CommitmentLoss.__name__,
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
        config = cls.codebook_build_pre_hook(config, registry, item)
        config = cls.commitment_build_pre_hook(config, registry, item)
        return config

    def forward(
        self,
        z: torch.Tensor,
        x: torch.Tensor,
        memo: Memo,
    ) -> torch.Tensor:
        codebook = self._codebook(z, x, memo)
        commitment = self._commitment(z, x, memo)
        return codebook + self._beta * commitment


@VQITQuantizerLossRegistry.register_()
class EntropyLoss(BaseLoss):  # TODO: refactor

    def __init__(self, *args, temperature: float, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._temperature = temperature

    def forward(
        self,
        z: torch.Tensor,
        x: torch.Tensor,
        memo: Memo,
    ) -> torch.Tensor:
        affinity: torch.Tensor = memo['distance']
        flat_affinity = affinity.reshape(-1, affinity.shape[-1])
        flat_affinity = flat_affinity / self._temperature
        probs = flat_affinity.softmax(-1)
        log_probs = torch.log_softmax(flat_affinity + 1e-5, -1)
        target_probs = probs
        avg_probs = target_probs.mean(0)
        avg_entropy = -torch.sum(avg_probs * torch.log(avg_probs + 1e-5))
        sample_entropy = -torch.mean(torch.sum(target_probs * log_probs, -1))
        loss = sample_entropy - avg_entropy
        return loss
