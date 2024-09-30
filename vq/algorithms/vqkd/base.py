# pylint: disable=duplicate-code

__all__ = [
    'VQKD',
]

from typing import Literal, cast

import todd
import todd.tasks.knowledge_distillation as kd
import torch
from todd.runners import Memo

from vq import VQModelRegistry
from vq.datasets import Batch
from vq.runners import BaseMixin as BaseRunnerMixin
from vq.tasks.image_reconstruction import BaseModel as BaseIRModel

from .registries import VQTeacherRegistry


@kd.KDDistillerRegistry.register_()
class VQKDDistiller(kd.distillers.SingleTeacherDistiller):

    @classmethod
    def build_pre_hook(
        cls,
        config: todd.Config,
        registry: todd.bases.registries.RegistryMeta,
        item: todd.bases.registries.Item,
    ) -> todd.Config:
        config = super().build_pre_hook(config, registry, item)
        config.teacher = VQTeacherRegistry.build_or_return(config.teacher)
        return config


@VQModelRegistry.register_()
class VQKD(kd.distillers.StudentMixin[VQKDDistiller], BaseIRModel):

    @kd.distillers.distiller_decorator
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def d_loss(
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
        runner: BaseRunnerMixin,
        batch: Batch,
        memo: Memo,
        *args,
        mode: Literal['train'] | None,
        **kwargs,
    ) -> Memo:
        log: Memo | None = memo.get('log')
        tensorboard: todd.runners.callbacks.TensorBoardCallback | None = \
            memo.get('tensorboard')

        original_image: torch.Tensor = batch['original_image']
        image: torch.Tensor = batch['image']
        if todd.Store.cuda:
            original_image = original_image.cuda()
            image = image.cuda()
        memo.update(original_image=original_image, image=image)

        self.distiller.teacher(original_image)

        x, memo = self.encode(image, memo)
        memo['x'] = x

        z, q_loss, memo = self.quantize(x, memo)
        memo['z'] = z

        pred_features, memo = self.decode(z, memo)
        memo['pred_features'] = pred_features

        d_loss, d_losses = self.d_loss(pred_features)

        losses = dict(**d_losses, q_loss=q_loss)
        if 'losses' in memo['quantizer']:
            losses.update(memo['quantizer']['losses'])
        memo['losses'] = losses

        memo['loss'] = q_loss + d_loss

        if mode == 'train':
            if log is not None:
                log.update({k: f'{v:.3f}' for k, v in losses.items()})
            if tensorboard is not None:
                for k, v in losses.items():
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
