__all__ = [
    'X2IMixin',
]

import enum
from typing import Literal, TypeVar

import todd.tasks.large_multimodal_model as lmm
import torch
from todd.runners import Memo
from torch import nn

from vq.datasets import Batch
from vq.tasks.sequence_modeling.models import X2I
from vq.tasks.sequence_modeling.runners import BaseMixin as BaseRunnerMixin
from vq.utils import get_memo

from .image import ImageMixin

T = TypeVar('T', bound=enum.Enum)
ModuleType = TypeVar('ModuleType', bound=nn.Module)


class X2IMixin(ImageMixin[T], X2I[T]):

    def sample(
        self,
        logits: torch.Tensor,
        memo: Memo,
    ) -> tuple[torch.Tensor, Memo]:
        start: int = memo['image_segment_start']
        image_wh: tuple[int, int] = memo['image_wh']
        codebook: lmm.Codebook[T] = memo['codebook']

        logits = logits[:, start - 1:-1]
        image_tokens, memo['transformer'] = self._transformer.sample(
            logits, 
            codebook, 
            get_memo(memo, 'transformer'),
        )
        image_tokens = codebook.debias(image_tokens)
        memo['image_tokens'] = image_tokens

        image, memo = self.decode_image_tokens(image_tokens, image_wh, memo)
        return image, memo

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
        image_segment = data.image_segment

        tokens = data.tokens[:, :image_segment.middle]
        if mode == 'train':
            tokens = tokens[:4]
        generate_memo: Memo = memo['generate']
        half_generate_memo = get_memo(memo, 'half_generate')
        for k, v in generate_memo.items():
            half_generate_memo.setdefault(k, v)
        half_generated_image, memo['half_generate'] = self.generate(
            tokens,
            half_generate_memo,
        )

        logits = memo['transformer']['logits']
        if mode == 'train':
            logits = logits[:4]
        sample_memo = get_memo(memo, 'sample')
        sample_memo.setdefault('image_segment_start', image_segment.start)
        sample_memo.setdefault('image_wh', data.image_wh)
        sample_memo.setdefault('codebook', data.image_codebook)
        sampled_image, memo['sample'] = self.sample(logits, sample_memo)

        images = dict(
            half_generated_image=half_generated_image,
            sampled_image=sampled_image,
        )
        memo.update(images)

        decoded_images = {
            k: runner.dataset.decode(v)
            for k, v in images.items()
        }
        if batched_visual is not None:
            batched_visual.update(decoded_images)
        if unbatched_visual is not None:
            unbatched_visual.update(decoded_images)

        return memo
