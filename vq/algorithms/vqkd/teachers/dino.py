__all__ = [
    'DINOTeacher',
]

import todd
import torch
from todd.bases.registries import Item
from todd.models.modules import DINO
from todd.registries import InitWeightsMixin

from vq.utils import Store

from ..registries import VQTeacherRegistry
from .base import BaseTeacher


@VQTeacherRegistry.register_()
class DINOTeacher(InitWeightsMixin, BaseTeacher[DINO]):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(
            *args,
            mean=todd.datasets.IMAGENET_MEAN_255,
            std=todd.datasets.IMAGENET_STD_255,
            **kwargs,
        )

    @classmethod
    def model_build_pre_hook(
        cls,
        config: todd.Config,
        registry: todd.RegistryMeta,
        item: Item,
    ) -> todd.Config:
        config = super().model_build_pre_hook(config, registry, item)
        model = config.model
        if isinstance(model, todd.Config):
            config.model = DINO(**model)
        return config

    @property
    def out_channels(self) -> int:
        return self._model.width

    def init_weights(self, config: todd.Config) -> bool:
        super().init_weights(config)
        self._model.load_pretrained(
            'pretrained/dino/vitbase16.pth',
            directory=Store.PRETRAINED,
        )
        return False

    def _forward(self, image: torch.Tensor, return_2d: bool) -> torch.Tensor:
        _, x = self._model(image, return_2d)
        return x
