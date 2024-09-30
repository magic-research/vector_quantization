__all__ = [
    'BaseTeacher',
]

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import einops
import todd
import torch
import torch.nn.functional as F
from todd.bases.registries import BuildPreHookMixin, Item
from todd.models import MeanStdMixin
from torch import nn

T = TypeVar('T', bound=nn.Module)


class BaseTeacher(BuildPreHookMixin, MeanStdMixin, ABC, Generic[T]):

    def __init__(
        self,
        *args,
        model: T,
        downsample_factor: int,
        image_wh: tuple[int, int] | None = None,
        output_wh: tuple[int, int] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if image_wh is None and output_wh is not None:
            image_wh = (
                output_wh[0] * downsample_factor,
                output_wh[1] * downsample_factor,
            )
        self._model = model
        self._downsample_factor = downsample_factor
        self._image_wh = image_wh
        self._output_wh = output_wh

    @classmethod
    def model_build_pre_hook(
        cls,
        config: todd.Config,
        registry: todd.RegistryMeta,
        item: Item,
    ) -> todd.Config:
        config.setdefault('model', todd.Config())
        return config

    @classmethod
    def build_pre_hook(
        cls,
        config: todd.Config,
        registry: todd.RegistryMeta,
        item: Item,
    ) -> todd.Config:
        config = super().build_pre_hook(config, registry, item)
        config = cls.model_build_pre_hook(config, registry, item)
        return config

    @property
    @abstractmethod
    def out_channels(self) -> int:
        pass

    @abstractmethod
    def _forward(self, image: torch.Tensor, return_2d: bool) -> torch.Tensor:
        pass

    def forward(
        self,
        original_image: torch.Tensor,
        return_2d: bool = False,
    ) -> torch.Tensor:
        image = self.normalize(original_image)

        if self._image_wh is not None:
            w, h = self._image_wh
            image = F.interpolate(image, (h, w), mode='bicubic')

        x = self._forward(image, return_2d or self._output_wh is not None)
        x = x.float()

        if self._output_wh is not None:
            w, h = self._output_wh
            x = F.interpolate(x, (h, w), mode='bicubic')

        if not return_2d and x.ndim != 3:
            x = einops.rearrange(x, 'b c h w -> b (h w) c')

        return x
