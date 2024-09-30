__all__ = [
    'VQITQuantizerDistanceRegistry',
    'BaseDistance',
    'L2Distance',
    'CosineDistance',
]

from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from torch import nn

from vq.tasks.image_tokenization.models import VQITQuantizerRegistry


class VQITQuantizerDistanceRegistry(VQITQuantizerRegistry):
    pass


class BaseDistance(nn.Module, ABC):

    @abstractmethod
    def forward(self, x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        pass


@VQITQuantizerDistanceRegistry.register_()
class L2Distance(BaseDistance):

    def forward(self, x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        return torch.cdist(x, e)


@VQITQuantizerDistanceRegistry.register_()
class CosineDistance(BaseDistance):

    @staticmethod
    def cosine_similarity(x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        # fixed CUDA out of memory error of torch.cosine_similarity
        x = F.normalize(x)
        e = F.normalize(e)
        return torch.einsum('x d, e d -> x e', x, e)

    def forward(self, x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        return 1 - self.cosine_similarity(x, e)
