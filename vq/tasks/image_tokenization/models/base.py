__all__ = [
    'BaseModel',
]

from abc import ABC, abstractmethod
from typing import Literal, TypeVar, cast

import einops
import todd
import torch
from todd.bases.registries import Item
from todd.runners import Memo
from torch import nn

from vq.datasets import Batch
from vq.models import VQEncoderRegistry
from vq.models.autoencoders import BaseEncoder
from vq.runners import BaseMixin as BaseRunnerMixin
from vq.utils import get_memo

from .connectors import BaseConnector, ComposedConnector
from .quantizers import BaseQuantizer
from .registries import VQITConnectorRegistry, VQITQuantizerRegistry

T = TypeVar('T', bound=nn.Module)


class BaseModel(todd.models.FreezeMixin, ABC):

    def __init__(
        self,
        *args,
        encoder: BaseEncoder,
        post_encode: BaseConnector,
        quantizer: BaseQuantizer,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._encoder = encoder
        self._post_encode = post_encode
        self._quantizer = quantizer

    @classmethod
    def encoder_build_pre_hook(
        cls,
        config: todd.Config,
        registry: todd.RegistryMeta,
        item: Item,
    ) -> todd.Config:
        config.encoder = VQEncoderRegistry.build_or_return(config.encoder)
        return config

    @classmethod
    def quantizer_build_pre_hook(
        cls,
        config: todd.Config,
        registry: todd.RegistryMeta,
        item: Item,
    ) -> todd.Config:
        config.quantizer = VQITQuantizerRegistry.build_or_return(
            config.quantizer,
        )
        return config

    @classmethod
    def post_encode_build_pre_hook(
        cls,
        config: todd.Config,
        registry: todd.RegistryMeta,
        item: Item,
    ) -> todd.Config:
        encoder: BaseEncoder = config.encoder
        quantizer: BaseQuantizer = config.quantizer
        post_encode = config.post_encode
        if isinstance(post_encode, list):
            post_encode = todd.Config(
                type=ComposedConnector.__name__,
                connectors=post_encode,
            )
        config.post_encode = VQITConnectorRegistry.build_or_return(
            post_encode,
            in_channels=encoder.out_channels,
            out_channels=quantizer.embedding_dim,
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
        config = cls.encoder_build_pre_hook(config, registry, item)
        config = cls.quantizer_build_pre_hook(config, registry, item)
        config = cls.post_encode_build_pre_hook(config, registry, item)
        return config

    @property
    def quantizer(self) -> BaseQuantizer:
        return self._quantizer

    def encode(
        self,
        image: torch.Tensor,
        memo: Memo,
    ) -> tuple[torch.Tensor, Memo]:
        x, memo['encoder'] = self._encoder(image, get_memo(memo, 'encoder'))
        x, memo['post_encode'] = self._post_encode(
            x,
            get_memo(memo, 'post_encode'),
        )
        return x, memo

    def quantize(
        self,
        x: torch.Tensor,
        memo: Memo,
    ) -> tuple[torch.Tensor, torch.Tensor, Memo]:
        quantizer_memo = get_memo(memo, 'quantizer')
        quantizer_memo['x_shape'] = (b, c, h, w) = x.shape

        x = einops.rearrange(x, 'b c h w -> (b h w) c')
        z, q_loss, memo['quantizer'] = self._quantizer(x, quantizer_memo)
        z = einops.rearrange(z, '(b h w) c -> b c h w', b=b, c=c, h=h, w=w)
        z = cast(torch.Tensor, z).contiguous()
        return z, q_loss, memo

    def encode_to_quant(
        self,
        image: torch.Tensor,
        memo: Memo,
    ) -> tuple[torch.Tensor, Memo]:
        x, memo = self.encode(image, memo)

        quantizer_memo = get_memo(memo, 'quantizer')
        quantizer_memo['x_shape'] = (b, _, h, w) = x.shape

        x = einops.rearrange(x, 'b c h w -> (b h w) c')
        x, quant, memo['quantizer'] = self._quantizer.encode(x, quantizer_memo)
        quantizer_memo = memo['quantizer']
        quantizer_memo.update(x=x, quant=quant)

        quant = einops.rearrange(quant, '(b h w) -> b h w', b=b, h=h, w=w)
        return quant, memo

    @property
    def codebook_size(self) -> int:
        return self._quantizer.codebook_size

    @abstractmethod
    def forward(
        self,
        runner: BaseRunnerMixin[T],
        batch: Batch,
        memo: Memo,
        *args,
        mode: Literal['train'] | None,
        **kwargs,
    ) -> Memo:
        pass
