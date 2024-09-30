__all__ = [
    'TorchVisionTeacher',
]

from typing import TypeVar

import todd
from todd.bases.registries import Item
from todd.models import TorchVisionRegistry
from torch import nn

from .base import BaseTeacher

T = TypeVar('T', bound=nn.Module)


class TorchVisionTeacher(BaseTeacher[T]):
    REGISTRY: type[TorchVisionRegistry]

    def __init__(
        self,
        *args,
        mean=todd.datasets.IMAGENET_MEAN_255,
        std=todd.datasets.IMAGENET_STD_255,
        **kwargs,
    ) -> None:
        super().__init__(*args, mean=mean, std=std, **kwargs)

    @classmethod
    def model_build_pre_hook(
        cls,
        config: todd.Config,
        registry: todd.RegistryMeta,
        item: Item,
    ) -> todd.Config:
        config.model = cls.REGISTRY.build_or_return(config.model)
        return config
