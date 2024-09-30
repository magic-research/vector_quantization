__all__ = [
    'CLIPTeacher',
]

import os

import clip.model
import todd
import torch
from todd.bases.registries import Item

from vq.utils import Store

from ..registries import VQTeacherRegistry
from .base import BaseTeacher


@VQTeacherRegistry.register_()
class CLIPTeacher(BaseTeacher[clip.model.CLIP]):

    def __init__(
        self,
        *args,
        with_proj: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            mean=todd.datasets.CLIP_MEAN_255,
            std=todd.datasets.CLIP_STD_255,
            **kwargs,
        )
        self._with_proj = with_proj

        if not with_proj:
            self.visual.proj = None

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
            model.setdefault('name', 'pretrained/clip/ViT-B-16.pt')
            model.name = os.path.join(Store.PRETRAINED, model.name)
            model, _ = clip.load(**model)
            config.model = model
        return config

    @property
    def visual(self) -> clip.model.VisionTransformer:
        return self._model.visual

    @property
    def out_channels(self) -> int:
        if self._with_proj:
            return self.visual.output_dim
        return self.visual.class_embedding.numel()

    def _forward(self, image: torch.Tensor, return_2d: bool) -> torch.Tensor:
        return self._model.encode_image(image, return_2d)
