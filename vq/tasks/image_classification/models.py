__all__ = [
    'BaseModel',
]

import enum
from typing import Generic, Literal, TypeVar, cast

import einops
import todd
import torch
from todd.bases.registries import Item
from todd.runners import Memo
from todd.runners.callbacks import TensorBoardCallback
from torch import nn

from vq.datasets import Batch
from vq.runners import BaseMixin as BaseRunnerMixin
from vq.tasks.image_tokenization.models import BaseModel as BaseITModel
from vq.utils import get_memo, load

from .registries import VQICModelRegistry

T = TypeVar('T', bound=enum.Enum)
ModuleType = TypeVar('ModuleType', bound=nn.Module)


@VQICModelRegistry.register_()
class BaseModel(todd.models.FreezeMixin, Generic[T]):

    def __init__(
        self,
        *args,
        it: BaseITModel,
        num_categories: int,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._it = it
        self._head = nn.Sequential(
            nn.BatchNorm1d(it.quantizer.embedding_dim, 1e-6, affine=False),
            nn.Linear(it.quantizer.embedding_dim, num_categories),
        )
        self._loss = nn.CrossEntropyLoss()  # TODO

    @classmethod
    def it_build_pre_hook(
        cls,
        config: todd.Config,
        registry: todd.RegistryMeta,
        item: Item,
    ) -> todd.Config:
        it = config.it
        if isinstance(it, todd.Config):
            config.it = load(**it)
        return config

    @classmethod
    def build_pre_hook(
        cls,
        config: todd.Config,
        registry: todd.RegistryMeta,
        item: Item,
    ) -> todd.Config:
        config = cls.it_build_pre_hook(config, registry, item)
        return super().build_pre_hook(config, registry, item)

    def init_weights(self, config: todd.Config) -> bool:
        super().init_weights(config)
        linear = cast(nn.Linear, self._head[1])
        nn.init.trunc_normal_(linear.weight, std=0.01)
        return False

    def forward(
        self,
        runner: BaseRunnerMixin[ModuleType],
        batch: Batch,
        memo: Memo,
        *args,
        mode: Literal['train'] | None,
        **kwargs,
    ) -> Memo:
        log: Memo | None = memo.get('log')
        tensorboard: TensorBoardCallback | None = memo.get('tensorboard')

        original_image = batch['original_image']
        image = batch['image']
        category = batch['category']
        if todd.Store.cuda:
            original_image = original_image.cuda()
            image = image.cuda()
            category = category.cuda()
        memo.update(
            original_image=original_image,
            image=image,
            category=category,
        )

        it_memo = get_memo(memo, 'it')
        encoder_memo = get_memo(it_memo, 'encoder')
        encoder_memo['original_image'] = memo['original_image']
        tokens, it_memo['encode_to_quant'] = self._it.encode_to_quant(
            image,
            get_memo(it_memo, 'encode_to_quant'),
        )
        z, it_memo['quantizer'] = self._it.quantizer.decode(
            tokens,
            get_memo(it_memo, 'quantizer'),
        )
        features = einops.reduce(z, 'b h w c -> b c', 'mean')
        logits = self._head(features)
        memo['logits'] = logits
        loss: torch.Tensor = self._loss(logits, category)
        memo['loss'] = loss.view(1)

        if mode == 'train':
            if log is not None:
                log['loss'] = f'{loss:.3f}'
            if tensorboard is not None:
                tensorboard.summary_writer.add_scalar(
                    tensorboard.tag('loss'),
                    loss.float(),
                    runner.iter_,
                )
        else:
            assert mode is None

        return memo
