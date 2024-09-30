__all__ = [
    'CosineEmbeddingLoss',
]

import todd
import torch
from todd.bases.registries import Item
from todd.models import losses

from vq.models import VQLossRegistry


@VQLossRegistry.register_()
class CosineEmbeddingLoss(losses.BaseLoss):

    def __init__(
        self,
        *args,
        cosine_embedding: losses.CosineEmbeddingLoss,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._cosine_embedding = cosine_embedding

    @classmethod
    def cosine_embedding_build_pre_hook(
        cls,
        config: todd.Config,
        registry: todd.RegistryMeta,
        item: Item,
    ) -> todd.Config:
        config.cosine_embedding = losses.CosineEmbeddingLoss(
            reduction='none',
            **config.cosine_embedding,
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
        config = cls.cosine_embedding_build_pre_hook(config, registry, item)
        return config

    def forward(  # pylint: disable=arguments-differ
        self,
        pred_image: torch.Tensor,
        image: torch.Tensor,
    ) -> torch.Tensor:
        assert pred_image.shape == image.shape
        shape = pred_image.shape
        pred_image = pred_image.flatten(0, -2)
        image = image.flatten(0, -2)
        target = pred_image.new_ones(pred_image.shape[0])
        loss: torch.Tensor = self._cosine_embedding(
            pred_image,
            image,
            target,
        )
        loss = loss.reshape(shape[:-1])
        return self._reduce(loss)
