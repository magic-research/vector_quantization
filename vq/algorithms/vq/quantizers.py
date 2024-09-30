__all__ = [
    'VectorQuantizer',
]

import todd
import torch
from todd.bases.registries import Item
from todd.registries import ModelRegistry
from todd.runners import Memo
from torch import nn

from vq.tasks.image_tokenization.models import VQITQuantizerRegistry
from vq.tasks.image_tokenization.models.quantizers import BaseQuantizer
from vq.tasks.image_tokenization.models.quantizers.utils import ste

from .distances import BaseDistance, VQITQuantizerDistanceRegistry


@VQITQuantizerRegistry.register_()
class VectorQuantizer(BaseQuantizer):

    def __init__(
        self,
        *args,
        embedding: nn.Embedding,
        distance: BaseDistance,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._embedding = embedding
        self._distance = distance

    @classmethod
    def embedding_build_pre_hook(
        cls,
        config: todd.Config,
        registry: todd.RegistryMeta,
        item: Item,
    ) -> todd.Config:
        config.embedding = ModelRegistry.build_or_return(config.embedding)
        return config

    @classmethod
    def distance_build_pre_hook(
        cls,
        config: todd.Config,
        registry: todd.RegistryMeta,
        item: Item,
    ) -> todd.Config:
        config.distance = VQITQuantizerDistanceRegistry.build_or_return(
            config.distance,
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
        config = cls.embedding_build_pre_hook(config, registry, item)
        config = cls.distance_build_pre_hook(config, registry, item)
        return config

    @property
    def embedding(self) -> nn.Embedding:
        return self._embedding

    @property
    def distance(self) -> BaseDistance:
        return self._distance

    @property
    def embedding_dim(self) -> int:
        return self._embedding.embedding_dim

    @property
    def codebook_size(self) -> int:
        return self._embedding.num_embeddings

    @property
    def embeddings(self) -> torch.Tensor:
        return self._embedding.weight.clone()

    def _init_weights(self, config: todd.Config) -> bool:
        func = todd.registries.InitRegistry.build(config)
        func(self._embedding.weight)
        return False

    def _encode(
        self,
        x: torch.Tensor,
        memo: Memo,
    ) -> tuple[torch.Tensor, Memo]:
        distance = self._distance(x, self.embeddings)
        memo['distance'] = distance
        quant = torch.argmin(distance, dim=-1)
        return quant, memo

    def _decode(
        self,
        quant: torch.Tensor,
        memo: Memo,
    ) -> tuple[torch.Tensor, Memo]:
        z = self._embedding(quant)
        return z, memo

    def forward(
        self,
        x: torch.Tensor,
        memo: Memo,
    ) -> tuple[torch.Tensor, torch.Tensor, Memo]:
        z, loss, memo = super().forward(x, memo)
        z = ste(z, memo['x'])
        return z, loss, memo
