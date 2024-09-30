__all__ = [
    'VQGAN_VQKD',
]

from typing import Literal, TypeVar, cast

import todd
import todd.tasks.knowledge_distillation as kd
import torch
from todd.bases.registries import Item
from todd.runners import Memo
from torch import nn

from vq import VQModelRegistry
from vq.algorithms.vqgan import VQGAN
from vq.algorithms.vqkd.base import VQKDDistiller
from vq.datasets import Batch
from vq.models import BaseDecoder, VQDecoderRegistry
from vq.runners import BaseMixin as BaseRunnerMixin
from vq.tasks.image_tokenization.models import VQITConnectorRegistry
from vq.tasks.image_tokenization.models.connectors import (
    BaseConnector,
    ComposedConnector,
)
from vq.tasks.image_tokenization.models.quantizers import BaseQuantizer
from vq.utils.misc import get_memo

T = TypeVar('T', bound=nn.Module)


@VQModelRegistry.register_()
class VQGAN_VQKD(  # noqa: N801 pylint: disable=invalid-name
    kd.distillers.StudentMixin[VQKDDistiller],
    VQGAN,
):

    @kd.distillers.distiller_decorator
    def __init__(
        self,
        *args,
        vqkd_pre_decode: BaseConnector,
        vqkd_decoder: BaseDecoder,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._vqkd_pre_decode = vqkd_pre_decode
        self._vqkd_decoder = vqkd_decoder

    @classmethod
    def vqkd_decoder_build_pre_hook(
        cls,
        config: todd.Config,
        registry: todd.RegistryMeta,
        item: Item,
    ) -> todd.Config:
        config.vqkd_decoder = VQDecoderRegistry.build_or_return(
            config.vqkd_decoder,
        )
        return config

    @classmethod
    def vqkd_pre_decode_build_pre_hook(
        cls,
        config: todd.Config,
        registry: todd.RegistryMeta,
        item: Item,
    ) -> todd.Config:
        vqkd_decoder: BaseDecoder = config.vqkd_decoder
        quantizer: BaseQuantizer = config.quantizer
        vqkd_pre_decode = config.vqkd_pre_decode
        if isinstance(vqkd_pre_decode, list):
            vqkd_pre_decode = todd.Config(
                type=ComposedConnector.__name__,
                connectors=vqkd_pre_decode,
            )
        config.vqkd_pre_decode = VQITConnectorRegistry.build_or_return(
            vqkd_pre_decode,
            in_channels=quantizer.embedding_dim,
            out_channels=vqkd_decoder.in_channels,
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
        config = cls.vqkd_decoder_build_pre_hook(config, registry, item)
        config = cls.vqkd_pre_decode_build_pre_hook(config, registry, item)
        return config

    def vqkd_decode(
        self,
        z: torch.Tensor,
        memo: Memo,
    ) -> tuple[torch.Tensor, Memo]:
        z, memo['vqkd_pre_decode'] = self._pre_decode(
            z,
            get_memo(memo, 'vqkd_pre_decode'),
        )
        pred_image, memo['vqkd_decoder'] = self._decoder(
            z,
            get_memo(memo, 'vqkd_decoder'),
        )
        return pred_image, memo

    def vqkd_d_loss(
        self,
        pred_features: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        custom_tensors = dict(pred_features=pred_features)
        d_losses: dict[str, torch.Tensor] = self.distiller(custom_tensors)
        d_loss = sum(d_losses.values())
        d_loss = cast(torch.Tensor, d_loss)
        return d_loss, d_losses

    def forward(
        self,
        runner: BaseRunnerMixin[T],
        batch: Batch,
        memo: Memo,
        *args,
        mode: Literal['generation', 'discrimination'] | None,  # type: ignore[override] # noqa: E501 pylint: disable=line-too-long
        **kwargs,
    ) -> Memo:
        memo = super().forward(runner, batch, memo, *args, mode=mode, **kwargs)
        if mode == 'discrimination':
            return memo

        log: Memo | None = memo.get('log')
        tensorboard: todd.runners.callbacks.TensorBoardCallback | None = \
            memo.get('tensorboard')

        original_image = memo['original_image']

        self.distiller.teacher(original_image)

        z = memo['z']

        pred_features, memo = self.decode(z, memo)
        memo['pred_features'] = pred_features

        d_loss, d_losses = self.vqkd_d_loss(pred_features)

        if 'loss' in memo:
            memo['loss'] = memo['loss'] + d_loss

        if mode == 'generation':
            if log is not None:
                log.update({k: f'{v:.3f}' for k, v in d_losses.items()})
            if tensorboard is not None:
                for k, v in d_losses.items():
                    tensorboard.summary_writer.add_scalar(
                        tensorboard.tag(k),
                        v.float(),
                        runner.iter_,
                    )
        else:
            assert mode is None

        self.distiller.reset()
        self.distiller.step()
        return memo
