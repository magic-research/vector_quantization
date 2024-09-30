__all__ = [
    'BaseDiscriminatorLoss',
    'VQGANDiscriminatorLoss',
    'R1GradientPenalty',
]

from abc import ABC, abstractmethod

import todd
import torch
import torch.nn.functional as F

from ..discriminators import BaseDiscriminator
from .registries import VQDiscriminatorLossRegistry


class BaseDiscriminatorLoss(todd.models.losses.BaseLoss, ABC):

    @abstractmethod
    def forward(  # pylint: disable=arguments-differ
        self,
        logits_fake: torch.Tensor,
        logits_real: torch.Tensor,
    ) -> torch.Tensor:
        pass


@VQDiscriminatorLossRegistry.register_()
class VQGANDiscriminatorLoss(BaseDiscriminatorLoss):

    def forward(
        self,
        logits_fake: torch.Tensor,
        logits_real: torch.Tensor,
    ) -> torch.Tensor:
        loss_fake = F.relu(1. + logits_fake)
        loss_real = F.relu(1. - logits_real)
        loss = (loss_fake + loss_real) / 2
        return self._reduce(loss)


class R1GradientPenalty(todd.models.losses.BaseLoss):
    """https://arxiv.org/abs/1801.04406."""

    def forward(  # pylint: disable=arguments-differ
        self,
        discriminator: BaseDiscriminator,
        image: torch.Tensor,
    ) -> torch.Tensor:
        image = image.clone().requires_grad_()

        training = {
            module: module.training
            for module in discriminator.modules()
        }
        discriminator.eval()
        logits_real = discriminator(image)
        for module, mode in training.items():
            module.training = mode

        gradients, = torch.autograd.grad(
            logits_real,
            image,
            torch.ones_like(logits_real),
            create_graph=True,
        )
        gradient_penalty: torch.Tensor = gradients.norm(2, (1, 2, 3))
        gradient_penalty = gradient_penalty.pow(2)
        return self._reduce(gradient_penalty)
