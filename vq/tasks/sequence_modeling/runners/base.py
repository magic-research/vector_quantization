__all__ = [
    'BaseMixin',
    'BaseTrainer',
    'BaseValidator',
]

from typing import TypeVar

import todd
from todd.bases.registries import Item
from torch import nn

from vq import VQModelRegistry
from vq.datasets import BaseMixin as BaseDatasetMixin
from vq.runners import BaseMixin as BaseMixin_
from vq.runners import BaseTrainer as BaseTrainer_
from vq.runners import BaseValidator as BaseValidator_

from ..registries import VQSMRunnerRegistry

T = TypeVar('T', bound=nn.Module)


class BaseMixin(BaseMixin_[T]):

    @classmethod
    def model_build_pre_hook(
        cls,
        config: todd.Config,
        registry: todd.RegistryMeta,
        item: Item,
    ) -> todd.Config:
        # ensure that dataset is built
        config = cls.dataset_build_pre_hook(config, registry, item)
        dataset: BaseDatasetMixin = config.dataset
        config.model = VQModelRegistry.build_or_return(
            config.model,
            num_categories=dataset.num_categories,
        )
        return config


@VQSMRunnerRegistry.register_()
class BaseTrainer(BaseMixin[T], BaseTrainer_[T]):
    pass


@VQSMRunnerRegistry.register_()
class BaseValidator(BaseMixin[T], BaseValidator_[T]):
    pass
