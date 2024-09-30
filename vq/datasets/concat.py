__all__ = [
    'ConcatDataset',
]

from typing import Any, Never

import todd
import torch
from todd.bases.registries import Item
from todd.datasets.access_layers import BaseAccessLayer
from todd.registries import DatasetRegistry

from ..registries import VQDatasetRegistry
from .base import BaseMixin


class PseudoAccessLayer(BaseAccessLayer[Any, Never]):

    def __getitem__(self, *args, **kwargs) -> Never:
        raise NotImplementedError

    def __setitem__(self, *args, **kwargs) -> Never:
        raise NotImplementedError

    def __delitem__(self, *args, **kwargs) -> Never:
        raise NotImplementedError

    def __len__(self, *args, **kwargs) -> Never:
        raise NotImplementedError

    def __iter__(self, *args, **kwargs) -> Never:
        raise NotImplementedError

    @property
    def exists(self) -> Never:
        raise NotImplementedError

    def touch(self, *args, **kwargs) -> Never:
        raise NotImplementedError


@VQDatasetRegistry.register_()
class ConcatDataset(BaseMixin[Any, Never], torch.utils.data.ConcatDataset):

    @classmethod
    def datasets_build_pre_hook(
        cls,
        config: todd.Config,
        registry: todd.RegistryMeta,
        item: Item,
    ) -> todd.Config:
        config = cls.transforms_build_pre_hook(config, registry, item)
        config.datasets = [
            DatasetRegistry.build_or_return(
                dataset,
                name=config.name,
                num_categories=config.num_categories,
                image_size=config.image_size,
                transforms=config.transforms,
            ) for dataset in config.datasets
        ]
        return config

    @classmethod
    def build_pre_hook(
        cls,
        config: todd.Config,
        registry: todd.RegistryMeta,
        item: Item,
    ) -> todd.Config:
        config = cls.datasets_build_pre_hook(config, registry, item)
        return super().build_pre_hook(config, registry, item)
