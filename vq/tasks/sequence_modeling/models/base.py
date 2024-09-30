__all__ = [
    'BaseModel',
]

import enum
from abc import abstractmethod
from typing import Any, Generic, Literal, TypeVar

import todd
import todd.tasks.large_multimodal_model as lmm
import torch
from todd.bases.registries import Item
from todd.runners import Memo
from todd.runners.callbacks import TensorBoardCallback
from torch import nn

from vq.datasets import Batch
from vq.utils import get_memo

from ..runners import BaseMixin as BaseRunnerMixin
from .registries import VQSMTransformerRegistry
from .transformers import BaseTransformer

T = TypeVar('T', bound=enum.Enum)
ModuleType = TypeVar('ModuleType', bound=nn.Module)


class BaseModel(todd.models.FreezeMixin, Generic[T]):

    def __init__(
        self,
        *args,
        num_categories: int,
        transformer: BaseTransformer,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._num_categories = num_categories
        self._transformer = transformer

    @classmethod
    def transformer_build_pre_hook(
        cls,
        config: todd.Config,
        registry: todd.RegistryMeta,
        item: Item,
    ) -> todd.Config:
        vocabulary_size = config.pop('vocabulary_size', None)
        config.transformer = VQSMTransformerRegistry.build_or_return(
            config.transformer,
            vocabulary_size=vocabulary_size,
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
        config = cls.transformer_build_pre_hook(config, registry, item)
        return config

    @property
    def transformer(self) -> BaseTransformer:
        return self._transformer

    @abstractmethod
    def _data(
        self,
        runner: BaseRunnerMixin[ModuleType],
        memo: Memo,
    ) -> tuple[lmm.InterleavedData[T], Memo]:
        pass

    @abstractmethod
    def generate(self, tokens: torch.Tensor, memo: Memo) -> tuple[Any, Memo]:
        pass

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
        batched_visual: Memo | None = memo.get('batched_visual')
        unbatched_visual: Memo | None = memo.get('unbatched_visual')

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

        data, memo = self._data(runner, memo)
        memo['data'] = data

        loss: torch.Tensor
        loss, memo['transformer'] = self._transformer(
            data,
            get_memo(memo, 'transformer'),
        )
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

        if batched_visual is not None:
            batched_visual['image'] = original_image
        if unbatched_visual is not None:
            unbatched_visual['image'] = original_image

        return memo
