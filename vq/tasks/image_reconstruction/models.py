__all__ = [
    'BaseModel',
]

from abc import ABC

import einops
import todd
import torch
from todd.bases.registries import Item
from todd.runners import Memo

from vq.models import VQDecoderRegistry
from vq.models.autoencoders import BaseDecoder
from vq.tasks.image_tokenization.models import BaseModel as BaseITModel
from vq.tasks.image_tokenization.models import VQITConnectorRegistry
from vq.tasks.image_tokenization.models.connectors import (
    BaseConnector,
    ComposedConnector,
)
from vq.tasks.image_tokenization.models.quantizers import BaseQuantizer
from vq.utils import get_memo


class BaseModel(BaseITModel, ABC):

    def __init__(
        self,
        *args,
        pre_decode: BaseConnector,
        decoder: BaseDecoder,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._pre_decode = pre_decode
        self._decoder = decoder

    @classmethod
    def decoder_build_pre_hook(
        cls,
        config: todd.Config,
        registry: todd.RegistryMeta,
        item: Item,
    ) -> todd.Config:
        config.decoder = VQDecoderRegistry.build_or_return(config.decoder)
        return config

    @classmethod
    def pre_decode_build_pre_hook(
        cls,
        config: todd.Config,
        registry: todd.RegistryMeta,
        item: Item,
    ) -> todd.Config:
        decoder: BaseDecoder = config.decoder
        quantizer: BaseQuantizer = config.quantizer
        pre_decode = config.pre_decode
        if isinstance(pre_decode, list):
            pre_decode = todd.Config(
                type=ComposedConnector.__name__,
                connectors=pre_decode,
            )
        config.pre_decode = VQITConnectorRegistry.build_or_return(
            pre_decode,
            in_channels=quantizer.embedding_dim,
            out_channels=decoder.in_channels,
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
        config = cls.decoder_build_pre_hook(config, registry, item)
        config = cls.pre_decode_build_pre_hook(config, registry, item)
        return config

    def decode(
        self,
        z: torch.Tensor,
        memo: Memo,
    ) -> tuple[torch.Tensor, Memo]:
        z, memo['pre_decode'] = self._pre_decode(
            z,
            get_memo(memo, 'pre_decode'),
        )
        pred_image, memo['decoder'] = self._decoder(
            z,
            get_memo(memo, 'decoder'),
        )
        return pred_image, memo

    def decode_from_quant(
        self,
        quant: torch.Tensor,
        memo: Memo,
    ) -> tuple[torch.Tensor, Memo]:
        z, memo['quantizer'] = self._quantizer.decode(
            quant,
            get_memo(memo, 'quantizer'),
        )
        z = einops.rearrange(z, 'b h w c -> b c h w')
        pred_image, memo = self.decode(z, memo)
        return pred_image, memo
