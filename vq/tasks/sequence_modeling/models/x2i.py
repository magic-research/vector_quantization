__all__ = [
    'X2I',
]

import enum
from abc import abstractmethod
from typing import Literal, TypeVar

import todd.tasks.large_multimodal_model as lmm
import torch
from todd.runners import Memo
from torch import nn

from vq.datasets import Batch
from vq.utils import get_memo

from ..runners import BaseMixin as BaseRunnerMixin
from .image import ImageModel

T = TypeVar('T', bound=enum.Enum)
ModuleType = TypeVar('ModuleType', bound=nn.Module)


class X2I(ImageModel[T]):

    def __init__(self, *args, cfg: float | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._cfg = cfg

    def generate(
        self,
        tokens: torch.Tensor,
        memo: Memo,
    ) -> tuple[torch.Tensor, Memo]:
        start: int = memo['image_segment_start']
        image_wh: tuple[int, int] = memo['image_wh']
        codebook: lmm.Codebook[T] = memo['codebook']

        assert tokens.shape[1] >= start

        w, h = image_wh
        tokens, memo = self._transformer.generate(
            tokens,
            start + w * h,
            codebook,
            memo,
        )
        image_tokens = tokens[:, start:]
        image_tokens = codebook.debias(image_tokens)
        memo['image_tokens'] = image_tokens

        image, memo = self.decode_image_tokens(image_tokens, image_wh, memo)
        return image, memo

    @abstractmethod
    def uncondition_tokens(
        self,
        condition_tokens: torch.Tensor,
    ) -> torch.Tensor:
        pass

    def dropout_tokens(
        self,
        condition_tokens: torch.Tensor,
        memo: Memo,
    ) -> tuple[torch.Tensor, Memo]:
        if not self.training:
            return condition_tokens, memo
        indices = torch.rand(condition_tokens.shape[0]) < self._cfg
        memo['indices'] = indices
        uncondition_tokens = self.uncondition_tokens(condition_tokens[indices])
        condition_tokens[indices] = uncondition_tokens
        return condition_tokens, memo

    def cfg_tokens(self, condition_tokens: torch.Tensor) -> torch.Tensor:
        uncondition_tokens = self.uncondition_tokens(condition_tokens)
        return torch.cat([uncondition_tokens, condition_tokens])

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

        data: lmm.X2IData[T] = memo['data']

        # During training, this tokens may be dropped out.
        # Therefore, visualizations during training may not be accurate.
        tokens = data.condition_segment.tokens
        if mode == 'train':
            tokens = tokens[:4]
        if self._cfg is not None:
            tokens = self.cfg_tokens(tokens)
        tokens = data.condition_codebook.bias_(tokens)
        generate_memo = get_memo(memo, 'generate')
        generate_memo.setdefault(
            'image_segment_start',
            data.image_segment.start,
        )
        generate_memo.setdefault('image_wh', data.image_wh)
        generate_memo.setdefault('codebook', data.image_codebook)
        image, memo['generate'] = self.generate(tokens, generate_memo)
        if self._cfg is not None:
            image, _ = image.chunk(2)
        memo.update(generated_image=image)

        decoded_image = runner.dataset.decode(image)
        if batched_visual is not None:
            batched_visual['generated_image'] = decoded_image
        if unbatched_visual is not None:
            unbatched_visual['generated_image'] = decoded_image

        return memo
