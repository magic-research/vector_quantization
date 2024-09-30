__all__ = [
    'BaseGeneratorLoss',
    'VQGANGeneratorLoss',
    'NonSaturatingLoss',
]

from abc import ABC, abstractmethod

import todd
import torch
import torch.nn.functional as F

from .registries import VQGeneratorLossRegistry


class BaseGeneratorLoss(todd.models.losses.BaseLoss, ABC):

    @abstractmethod
    def forward(  # pylint: disable=arguments-differ
        self,
        logits_fake: torch.Tensor,
    ) -> torch.Tensor:
        pass


@VQGeneratorLossRegistry.register_()
class VQGANGeneratorLoss(BaseGeneratorLoss):

    def forward(self, logits_fake: torch.Tensor) -> torch.Tensor:
        loss = -logits_fake
        return self._reduce(loss)


@VQGeneratorLossRegistry.register_()
class NonSaturatingLoss(BaseGeneratorLoss):

    def forward(self, logits_fake: torch.Tensor) -> torch.Tensor:
        loss = F.binary_cross_entropy_with_logits(
            logits_fake,
            torch.ones_like(logits_fake),
        )
        return self._reduce(loss)
