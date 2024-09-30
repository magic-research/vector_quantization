__all__ = [
    'ViTEncoder',
    'ViTDecoder',
]

from typing import cast

import einops
import todd
import torch
import torchvision
from todd.bases.registries import BuildPreHookMixin, Item
from todd.models import ViTRegistry
from todd.runners import Memo
from torch import nn

from vq.models import (
    BaseDecoder,
    BaseEncoder,
    VQDecoderRegistry,
    VQEncoderRegistry,
)


class ViTMixin(BuildPreHookMixin, nn.Module):

    def __init__(
        self,
        *args,
        vit: torchvision.models.VisionTransformer,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._vit = vit

    @classmethod
    def vit_build_pre_hook(
        cls,
        config: todd.Config,
        registry: todd.RegistryMeta,
        item: Item,
    ) -> todd.Config:
        config.vit = ViTRegistry.build_or_return(config.vit)
        return config

    @classmethod
    def build_pre_hook(
        cls,
        config: todd.Config,
        registry: todd.RegistryMeta,
        item: Item,
    ) -> todd.Config:
        config = super().build_pre_hook(config, registry, item)
        config = cls.vit_build_pre_hook(config, registry, item)
        return config


@VQEncoderRegistry.register_()
class ViTEncoder(ViTMixin, BaseEncoder):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._vit.heads = None
        self._feat_size = self._vit.image_size // self._vit.patch_size

    @property
    def out_channels(self) -> int:
        return self._vit.hidden_dim

    def forward(
        self,
        image: torch.Tensor,
        memo: Memo,
    ) -> tuple[torch.Tensor, Memo]:
        b, _, h, w = image.shape
        assert h == w == self._vit.image_size
        x: torch.Tensor = self._vit.conv_proj(image)
        x = einops.rearrange(x, 'b c h w -> b (h w) c')
        class_token = einops.repeat(
            self._vit.class_token,
            '1 1 c -> b 1 c',
            b=b,
        )
        x = torch.cat([class_token, x], dim=1)
        x = self._vit.encoder(x)
        x = x[:, 1:]
        x = einops.rearrange(
            x,
            'b (h w) c -> b c h w',
            h=self._feat_size,
            w=self._feat_size,
        )
        x = x.contiguous()
        return x, memo


@VQDecoderRegistry.register_()
class ViTDecoder(ViTMixin, BaseDecoder):

    def __init__(self, *args, patch_size: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._vit.conv_proj = None
        self._vit.heads = None

        self._patch_size = patch_size

        hidden_dim = self._vit.hidden_dim
        self._out_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 3 * patch_size**2),
        )

    @classmethod
    def vit_build_pre_hook(
        cls,
        config: todd.Config,
        registry: todd.RegistryMeta,
        item: Item,
    ) -> todd.Config:
        config.vit = ViTRegistry.build_or_return(config.vit, patch_size=1)
        return config

    @property
    def in_channels(self) -> int:
        return self._vit.hidden_dim

    @property
    def last_parameter(self) -> nn.Parameter:
        last_layer: nn.Linear = self._out_proj[-1]
        return cast(nn.Parameter, last_layer.weight)

    def forward(
        self,
        z: torch.Tensor,
        memo: Memo,
    ) -> tuple[torch.Tensor, Memo]:
        b, _, h, w = z.shape
        assert h == w == self._vit.image_size
        x: torch.Tensor = einops.rearrange(z, 'b c h w -> b (h w) c')
        class_token = einops.repeat(
            self._vit.class_token,
            '1 1 c -> b 1 c',
            b=b,
        )
        x = torch.cat([class_token, x], dim=1)
        x = self._vit.encoder(x)
        x = x[:, 1:]
        x = self._out_proj(x)
        x = einops.rearrange(
            x,
            'b (h w) (p q c) -> b c (h p) (w q)',
            h=self._vit.image_size,
            w=self._vit.image_size,
            p=self._patch_size,
            q=self._patch_size,
            c=3,
        )
        x = x.contiguous()
        return x, memo
