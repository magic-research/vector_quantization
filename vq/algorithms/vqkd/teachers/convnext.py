__all__ = [
    'ConvNeXtTeacher',
]

from typing import cast

import einops
import torch
from todd.models import ConvNeXtRegistry
from torch import nn
from torchvision import models

from ..registries import VQTeacherRegistry
from .torchvision import TorchVisionTeacher


@VQTeacherRegistry.register_()
class ConvNeXtTeacher(TorchVisionTeacher[models.ConvNeXt]):
    REGISTRY = ConvNeXtRegistry

    @property
    def out_channels(self) -> int:
        layer_norm = cast(nn.LayerNorm, self._model.classifier[0])
        return layer_norm.normalized_shape[0]

    def _forward(self, image: torch.Tensor, return_2d: bool) -> torch.Tensor:
        x = self._model.features(image)
        if not return_2d:
            x = einops.rearrange(x, 'b c h w -> b (h w) c')
        return x
