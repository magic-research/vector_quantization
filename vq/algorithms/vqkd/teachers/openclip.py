__all__ = [
    'OpenCLIPTeacher',
]

import os

import einops
import open_clip
import todd
import torch
from open_clip.transformer import VisionTransformer
from todd.bases.registries import Item

from vq.utils import Store

from ..registries import VQTeacherRegistry
from .base import BaseTeacher

# TODO: copy openclip here


@VQTeacherRegistry.register_()
class OpenCLIPTeacher(BaseTeacher[open_clip.CLIP]):

    def __init__(self, *args, model: open_clip.CLIP, **kwargs) -> None:
        super().__init__(
            *args,
            mean=model.visual.image_mean,
            std=model.visual.image_std,
            model=model,
            **kwargs,
        )
        self.visual.pool_type = 'none'

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
            model.setdefault('cache_dir', 'pretrained/openclip')
            model.cache_dir = os.path.join(Store.PRETRAINED, model.cache_dir)
            model = open_clip.create_model(**model, require_pretrained=True)
            config.model = model
        return config

    @property
    def visual(self) -> VisionTransformer:
        return self._model.visual

    @property
    def out_channels(self) -> int:
        return self.visual.output_dim

    def _forward(self, image: torch.Tensor, return_2d: bool) -> torch.Tensor:
        x = self._model.encode_image(image)
        x = x[:, 1:]
        if return_2d:
            image_h, image_w = self.visual.preprocess_cfg['size']
            patch_h, patch_w = self.visual.patch_size
            h = image_h // patch_h
            w = image_w // patch_w
            x = einops.rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        return x
