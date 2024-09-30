__all__ = [
    'EVACLIPTeacher',
]

import os
from typing import TYPE_CHECKING

import todd
import torch
from todd.bases.registries import Item

from vq.utils import Store

from ..registries import VQTeacherRegistry
from .base import BaseTeacher

if TYPE_CHECKING:
    import eva_clip.eva_vit_model


@VQTeacherRegistry.register_()
class EVACLIPTeacher(BaseTeacher['eva_clip.CLIP']):

    def __init__(self, *args, model: 'eva_clip.CLIP', **kwargs) -> None:
        super().__init__(
            *args,
            mean=model.visual.image_mean,
            std=model.visual.image_std,
            model=model,
            **kwargs,
        )

    @classmethod
    def model_build_pre_hook(
        cls,
        config: todd.Config,
        registry: todd.RegistryMeta,
        item: Item,
    ) -> todd.Config:
        from eva_clip import (  # pylint: disable=import-outside-toplevel
            create_model_from_pretrained,
        )
        model = config.get('model', todd.Config())
        if isinstance(model, todd.Config):
            model.setdefault('cache_dir', 'pretrained/evaclip')
            model.cache_dir = os.path.join(Store.PRETRAINED, model.cache_dir)
            model = create_model_from_pretrained(
                **model,
                is_frozen=True,
                return_transform=False,
            )
            config.model = model
        return super().model_build_pre_hook(config, registry, item)

    @property
    def visual(self) -> 'eva_clip.eva_vit_model.EVAVisionTransformer':
        return self._model.visual

    @property
    def out_channels(self) -> int:
        return self._model.visual.output_dim

    def _forward(self, image: torch.Tensor, return_2d: bool) -> torch.Tensor:
        return self.visual.forward_features(image, return_2d=return_2d)
