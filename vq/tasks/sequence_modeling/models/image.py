__all__ = [
    'ImageModel',
]

import enum
from typing import Literal, TypeVar

import einops
import todd
import todd.tasks.large_multimodal_model as lmm
import torch
from todd.bases.registries import Item
from todd.runners import Memo
from torch import nn

from vq.datasets import Batch
from vq.tasks.image_reconstruction import BaseModel as BaseIRModel
from vq.tasks.image_tokenization.models.quantizers import BaseQuantizer
from vq.utils import get_memo, load

from ..runners import BaseMixin as BaseRunnerMixin
from .base import BaseModel

T = TypeVar('T', bound=enum.Enum)
ModuleType = TypeVar('ModuleType', bound=nn.Module)


class ImageModel(BaseModel[T]):

    def __init__(self, *args, ir: BaseIRModel, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._ir = ir

    @classmethod
    def ir_build_pre_hook(
        cls,
        config: todd.Config,
        registry: todd.RegistryMeta,
        item: Item,
    ) -> todd.Config:
        ir = config.ir
        if isinstance(ir, todd.Config):
            config.ir = load(**ir)
        return config

    @classmethod
    def build_pre_hook(
        cls,
        config: todd.Config,
        registry: todd.RegistryMeta,
        item: Item,
    ) -> todd.Config:
        config = cls.ir_build_pre_hook(config, registry, item)
        return super().build_pre_hook(config, registry, item)

    @property
    def quantizer(self) -> BaseQuantizer:
        return self._ir.quantizer

    def reconstruct_image(
        self,
        memo: Memo,
        mode: Literal['train'] | None = None,
    ) -> tuple[torch.Tensor, Memo]:
        data: lmm.ImageData[T] = memo['data']
        image_tokens = memo['image_tokens']
        if mode == 'train':
            image_tokens = image_tokens[:4]
        return self.decode_image_tokens(image_tokens, data.image_wh, memo)

    def forward(
        self,
        runner: BaseRunnerMixin[ModuleType],
        batch: Batch,
        memo: Memo,
        *args,
        mode: Literal['train'] | None,
        **kwargs,
    ) -> Memo:
        memo = super().forward(runner, batch, memo, *args, mode=mode, **kwargs)
        batched_visual: Memo | None = memo.get('batched_visual')
        unbatched_visual: Memo | None = memo.get('unbatched_visual')

        if (
            batched_visual is None and unbatched_visual is None
            and mode is not None
        ):
            return memo

        image, memo = self.reconstruct_image(memo, mode)
        memo['reconstructed_image'] = image

        decoded_image = runner.dataset.decode(image)
        if batched_visual is not None:
            batched_visual['reconstructed_image'] = decoded_image
        if unbatched_visual is not None:
            unbatched_visual['reconstructed_image'] = decoded_image

        return memo

    def encode_image_tokens(
        self,
        image: torch.Tensor,
        memo: Memo,
    ) -> tuple[torch.Tensor, Memo]:
        ir_memo = get_memo(memo, 'ir')
        encoder_memo = get_memo(ir_memo, 'encoder')
        encoder_memo['original_image'] = memo['original_image']
        tokens, ir_memo['encode_to_quant'] = self._ir.encode_to_quant(
            image,
            get_memo(ir_memo, 'encode_to_quant'),
        )
        memo['image_tokens'] = tokens
        return tokens, memo

    def decode_image_tokens(
        self,
        tokens: torch.Tensor,
        image_wh: tuple[int, int],
        memo: Memo,
    ) -> tuple[torch.Tensor, Memo]:
        ir_memo = get_memo(memo, 'ir')
        if tokens.ndim == 2:
            tokens = einops.rearrange(
                tokens,
                'b (h w) -> b h w',
                h=image_wh[1],
                w=image_wh[0],
            )
        elif tokens.ndim == 3:
            _, h, w = tokens.shape
            assert (w, h) == image_wh
        else:
            raise ValueError(f"Invalid tokens shape: {tokens.shape}")
        image, ir_memo['decode_from_quant'] = self._ir.decode_from_quant(
            tokens,
            get_memo(ir_memo, 'decode_from_quant'),
        )
        return image, memo
