__all__ = [
    'Cluster',
]

from typing import Literal, TypeVar

import todd
import torch
from todd.runners import Memo
from todd.runners.callbacks import TensorBoardCallback
from torch import nn

from vq import VQModelRegistry
from vq.datasets import Batch
from vq.runners import BaseMixin as BaseRunnerMixin
from vq.tasks.image_tokenization.models import BaseModel as BaseITModel
from vq.utils import get_memo

T = TypeVar('T', bound=nn.Module)


@VQModelRegistry.register_()
class Cluster(BaseITModel):

    def forward(
        self,
        runner: BaseRunnerMixin[T],
        batch: Batch,
        memo: Memo,
        *args,
        mode: Literal['train'] | None,
        **kwargs,
    ) -> Memo:
        log: Memo | None = memo.get('log')
        tensorboard: TensorBoardCallback[T] | None = memo.get('tensorboard')

        original_image: torch.Tensor = batch['original_image']
        image: torch.Tensor = batch['image']
        if todd.Store.cuda:
            original_image = original_image.cuda()
            image = image.cuda()
        memo.update(original_image=original_image, image=image)

        encoder_memo = get_memo(memo, 'encoder')
        encoder_memo['original_image'] = original_image
        x, memo = self.encode(image, memo)
        memo['x'] = x

        z, q_loss, memo = self.quantize(x, memo)
        memo.update(z=z, loss=q_loss)

        losses = dict(q_loss=q_loss)
        if 'loss' in memo['quantizer']:
            losses.update(memo['quantizer']['loss'])
        if log is not None:
            log.update({k: f'{v:.3f}' for k, v in losses.items()})
        if tensorboard is not None:
            for k, v in losses.items():
                tensorboard.summary_writer.add_scalar(
                    tensorboard.tag(k),
                    v.float(),
                    runner.iter_,
                )

        return memo
