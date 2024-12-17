__all__ = [
    'VQGAN',
]

from typing import Literal, TypeVar, cast

import todd
import torch
from todd.bases.registries import Item
from todd.patches.torch import ModuleDict
from todd.runners import Memo
from todd.runners.callbacks import TensorBoardCallback
from torch import nn

from vq import VQModelRegistry
from vq.datasets import Batch
from vq.runners import BaseMixin as BaseRunnerMixin
from vq.tasks.image_reconstruction import BaseModel as BaseIRModel
from vq.tasks.image_reconstruction import VQIRLossRegistry
from vq.utils import build_module_dict, get_memo

from .discriminators import BaseDiscriminator
from .losses import (
    BaseDiscriminatorLoss,
    BaseGeneratorLoss,
    R1GradientPenalty,
    VQDiscriminatorLossRegistry,
    VQGeneratorLossRegistry,
)
from .registries import VQDiscriminatorRegistry
from .trainer import VQGANTrainer

T = TypeVar('T', bound=nn.Module)


@VQModelRegistry.register_()
class VQGAN(BaseIRModel):

    def __init__(
        self,
        *args,
        discriminator: BaseDiscriminator,
        reconstruct_losses: ModuleDict,
        generator_loss: BaseGeneratorLoss,
        discriminator_loss: BaseDiscriminatorLoss,
        adaptive_generator_loss_weight_gain: float | None = 0.8,
        r1_gradient_penalty: R1GradientPenalty | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._discriminator = discriminator
        self._r_losses = reconstruct_losses
        self._g_loss = generator_loss
        self._d_loss = discriminator_loss
        self._aglwg = adaptive_generator_loss_weight_gain
        if r1_gradient_penalty is not None:
            self._r1_gp = r1_gradient_penalty

    @classmethod
    def discriminator_build_pre_hook(
        cls,
        config: todd.Config,
        registry: todd.RegistryMeta,
        item: Item,
    ) -> todd.Config:
        config.discriminator = VQDiscriminatorRegistry.build_or_return(
            config.discriminator,
        )
        return config

    @classmethod
    def reconstruct_losses_build_pre_hook(
        cls,
        config: todd.Config,
        registry: todd.RegistryMeta,
        item: Item,
    ) -> todd.Config:
        config.reconstruct_losses = build_module_dict(
            VQIRLossRegistry,
            config.reconstruct_losses,
        )
        return config

    @classmethod
    def generator_loss_build_pre_hook(
        cls,
        config: todd.Config,
        registry: todd.RegistryMeta,
        item: Item,
    ) -> todd.Config:
        config.generator_loss = VQGeneratorLossRegistry.build_or_return(
            config.generator_loss,
        )
        return config

    @classmethod
    def discriminator_loss_build_pre_hook(
        cls,
        config: todd.Config,
        registry: todd.RegistryMeta,
        item: Item,
    ) -> todd.Config:
        config.discriminator_loss = \
            VQDiscriminatorLossRegistry.build_or_return(
                config.discriminator_loss,
            )
        return config

    @classmethod
    def r1_gradient_penalty_build_pre_hook(
        cls,
        config: todd.Config,
        registry: todd.RegistryMeta,
        item: Item,
    ) -> todd.Config:
        r1_gradient_penalty = config.get('r1_gradient_penalty')
        if r1_gradient_penalty is None:
            return config
        config.r1_gradient_penalty = R1GradientPenalty(**r1_gradient_penalty)
        return config

    @classmethod
    def build_pre_hook(
        cls,
        config: todd.Config,
        registry: todd.RegistryMeta,
        item: Item,
    ) -> todd.Config:
        config = super().build_pre_hook(config, registry, item)
        config = cls.discriminator_build_pre_hook(config, registry, item)
        config = cls.reconstruct_losses_build_pre_hook(config, registry, item)
        config = cls.generator_loss_build_pre_hook(config, registry, item)
        config = cls.discriminator_loss_build_pre_hook(config, registry, item)
        config = cls.r1_gradient_penalty_build_pre_hook(config, registry, item)
        return config

    @property
    def with_r1_gp(self) -> bool:
        return hasattr(self, '_r1_gp')

    def _aglw(
        self,
        r_loss: torch.Tensor,
        g_loss: torch.Tensor,
    ) -> torch.Tensor:
        if self._aglwg is None:
            return r_loss.new_ones([])
        last_parameter = self._decoder.last_parameter
        r_grads, = torch.autograd.grad(
            r_loss,
            last_parameter,
            retain_graph=True,
        )
        g_grads, = torch.autograd.grad(
            g_loss,
            last_parameter,
            retain_graph=True,
        )

        aglw = torch.norm(r_grads) / (torch.norm(g_grads) + 1e-4)
        aglw = torch.clamp(aglw, 0.0, 1e4).detach()
        aglw = aglw * self._aglwg
        return aglw

    def r_loss(
        self,
        pred_image: torch.Tensor,
        image: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        r_losses = {k: v(pred_image, image) for k, v in self._r_losses.items()}
        r_loss = sum(r_losses.values())
        r_loss = cast(torch.Tensor, r_loss)
        return r_loss, r_losses

    def g_loss(
        self,
        pred_image: torch.Tensor,
        r_loss: torch.Tensor,
        with_d: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if with_d:
            logits_fake = self._discriminator(pred_image)
            g_loss = self._g_loss(logits_fake)
            aglw = self._aglw(r_loss, g_loss)
        else:
            g_loss = aglw = pred_image.new_zeros([])
        return g_loss, aglw

    def d_loss(
        self,
        pred_image: torch.Tensor,
        image: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert not pred_image.requires_grad
        logits_fake = self._discriminator(pred_image)
        logits_real = self._discriminator(image)
        d_loss = self._d_loss(logits_fake, logits_real)
        r1_gp = (
            self._r1_gp(self._discriminator, image)
            if self.with_r1_gp else image.new_zeros([])
        )
        return d_loss, r1_gp

    def forward(  # noqa: C901
        self,
        runner: BaseRunnerMixin[T],
        batch: Batch,
        memo: Memo,
        *args,
        mode: Literal['generation', 'discrimination'] | None,  # type: ignore[override] # noqa: E501 pylint: disable=line-too-long
        **kwargs,
    ) -> Memo:
        log: Memo | None = memo.get('log')
        tensorboard: TensorBoardCallback | None = memo.get('tensorboard')
        batched_visual: Memo | None = memo.get('batched_visual')
        unbatched_visual: Memo | None = memo.get('unbatched_visual')

        original_image = batch['original_image']
        image = batch['image']
        if todd.Store.cuda:
            original_image = original_image.cuda()
            image = image.cuda()
        memo.update(original_image=original_image, image=image)

        encoder_memo = get_memo(memo, 'encoder')
        encoder_memo['original_image'] = original_image
        x, memo = self.encode(image, memo)
        memo['x'] = x

        z, q_loss, memo = self.quantize(x, memo)
        memo['z'] = z

        pred_image, memo = self.decode(z, memo)
        memo['pred_image'] = pred_image

        if mode in ['generation', None]:
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

            images = dict(image=image, pred_image=pred_image)
            decoded_images = {
                k: runner.dataset.decode(v)
                for k, v in images.items()
            }
            if batched_visual is not None:
                batched_visual.update(decoded_images)
            if unbatched_visual is not None:
                unbatched_visual.update(decoded_images)

        if mode == 'generation':
            r_loss, r_losses = self.r_loss(pred_image, image)
            g_loss, aglw = self.g_loss(
                pred_image,
                r_loss,
                cast(VQGANTrainer[T], runner).with_d,
            )
            tensors = dict(**r_losses, g_loss=g_loss, aglw=aglw)
            memo['loss'] = q_loss + r_loss + g_loss * aglw
        elif mode == 'discrimination':
            d_loss, r1_gp = self.d_loss(
                pred_image,
                image,
            )
            tensors = dict(d_loss=d_loss)
            if self.with_r1_gp:
                tensors['r1_gp'] = r1_gp
            memo['loss'] = d_loss + r1_gp
        else:
            assert mode is None
            tensors = dict()

        if log is not None:
            log.update({k: f'{v:.3f}' for k, v in tensors.items()})
        if tensorboard is not None:
            for k, v in tensors.items():
                tensorboard.summary_writer.add_scalar(
                    tensorboard.tag(k),
                    v.float(),
                    runner.iter_,
                )

        return memo
