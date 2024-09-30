__all__ = [
    'ViTTeacher',
]

import einops
import torch
import torch.nn.functional as F
from todd.models import ViTRegistry
from torch import nn
from torchvision import models

from ..registries import VQTeacherRegistry
from .torchvision import TorchVisionTeacher

# TODO: refactor


def forward_hook(
    module: models.vision_transformer.Encoder,
    inputs: tuple[torch.Tensor],
    output: torch.Tensor,
) -> torch.Tensor:
    output = einops.rearrange(output[:, :-1], 'b l n -> b 1 l n')
    return output


@VQTeacherRegistry.register_()
class ViTTeacher(TorchVisionTeacher[models.VisionTransformer]):
    REGISTRY = ViTRegistry

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        model = self._model

        model.encoder.register_forward_hook(forward_hook)
        model.heads = nn.Identity()
        self._model = model

        image_size = model.image_size
        patch_size = model.patch_size
        grid_size = image_size // patch_size
        self._grid_size = grid_size

        position_embedding = model.encoder.pos_embedding
        self._position_embedding = position_embedding

        self._position_embeddings = {
            image_size: position_embedding,
        }

    @property
    def out_channels(self) -> int:
        return self._model.hidden_dim

    def _interpolate_position_embedding(
        self,
        image_size: int,
    ) -> torch.Tensor:
        # TODO: checkout torchvision.models.interpolate_embeddings
        if image_size in self._position_embeddings:
            return self._position_embeddings[image_size]
        position_embedding = self._position_embedding[:, 1:]
        position_embedding = einops.rearrange(
            position_embedding,
            '1 (h w) c -> 1 c h w',
            h=self._grid_size,
            w=self._grid_size,
        )
        # TODO: use _grid_wh instead
        assert image_size % self._model.patch_size == 0
        grid_size = image_size // self._model.patch_size
        position_embedding = F.interpolate(
            position_embedding,
            (grid_size, grid_size),
        )
        position_embedding = einops.rearrange(
            position_embedding,
            '1 c h w -> 1 (h w) c',
        )
        position_embedding = torch.cat(
            [
                self._position_embedding[:, [0]],
                position_embedding,
            ],
            1,
        )
        position_embedding = nn.Parameter(
            position_embedding,
            requires_grad=position_embedding.requires_grad,
        )
        self._position_embeddings[image_size] = position_embedding
        return position_embedding

    def _forward(self, image: torch.Tensor, return_2d: bool) -> torch.Tensor:
        x = self._model(image)
        if return_2d:
            with torch.device('meta'):
                module = self._model.conv_proj.to('meta')
                x_: torch.Tensor = module(image.to('meta'))
            _, _, h, w = x_.shape
            x = einops.rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        return x

    def forward(
        self,
        original_image: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        _, _, h, w = original_image.shape
        assert h == w
        self._model.image_size = h
        position_embedding = self._interpolate_position_embedding(h)
        self._model.encoder.pos_embedding = position_embedding
        return super().forward(original_image, *args, **kwargs)
