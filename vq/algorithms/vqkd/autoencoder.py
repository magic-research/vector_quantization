# TODO: refactor
# flake8: noqa

__all__ = [
    'VQKDEncoder',
    'VQKDDecoder',
]

import math
from typing import cast

import einops
import todd
import torch
import torch.nn.functional as F
from todd.runners import Memo
from torch import nn

from vq.models import VQDecoderRegistry, VQEncoderRegistry
from vq.models.autoencoders import BaseDecoder, BaseEncoder


class Mlp(nn.Module):

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # commit this for the original BERT implement
        x = self.fc2(x)
        return x


class Attention(nn.Module):

    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        all_head_dim = head_dim * self.num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
        self.v_bias = nn.Parameter(torch.zeros(all_head_dim))

        self.proj = nn.Linear(all_head_dim, dim)

    def forward(self, x):
        b, n, _ = x.shape
        qkv_bias = torch.cat((
            self.q_bias, torch.zeros_like(self.v_bias,
                                          requires_grad=False), self.v_bias
        ))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(b, n, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[
            2
        ]  # make torchscript happy (cannot use tensor as tuple) (B, H, N, C)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(b, n, -1)
        x = self.proj(x)

        return x


class Block(nn.Module):

    def __init__(
        self,
        dim,
        num_heads,
        norm_layer: todd.Config,
        mlp_ratio=4.,
    ):
        super().__init__()
        self.norm1 = todd.models.NormRegistry.build(
            norm_layer,
            normalized_shape=dim,
        )
        self.attn = Attention(
            dim,
            num_heads=num_heads,
        )
        self.norm2 = todd.models.NormRegistry.build(
            norm_layer,
            normalized_shape=dim,
        )
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding."""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size)**2
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class VQKDMixin(nn.Module):

    def __init__(
        self,
        *args,
        img_size: int,
        patch_size: int,
        in_chans: int,
        out_chans: int,
        out_patch_size: int = 1,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        norm_layer = todd.Config(type='LN', eps=1e-06)
        self.num_features = self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.out_patch_size = out_patch_size

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim)
        )

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                norm_layer=norm_layer
            ) for i in range(depth)
        ])
        self.fc_norm = nn.LayerNorm(embed_dim, 1e-6)

        self.task_layer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, out_chans * out_patch_size**2),
        )

    @staticmethod
    def _init_weights(m) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            return
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)
            return

    def init_weights(self, config: todd.Config) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)
        self.blocks.apply(self._init_weights)
        self.fc_norm.apply(self._init_weights)
        for i, block in enumerate(self.blocks):
            block = cast(Block, block)
            divisor = (2 * (i + 1))**0.5
            block.attn.proj.weight.data.div_(divisor)
            block.mlp.fc2.weight.data.div_(divisor)
        self.task_layer.apply(self._init_weights)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        n = self.pos_embed.shape[1] - 1
        if npatch == n and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the
        # interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(
                1, int(math.sqrt(n)), int(math.sqrt(n)), dim
            ).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(n), h0 / math.sqrt(n)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[
            -2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed),
                         dim=1)

    def forward(
        self,
        x: torch.Tensor,
        memo: Memo,
    ) -> tuple[torch.Tensor, Memo]:
        _, _, w, h = x.shape
        x = self.patch_embed(x)
        batch_size, _, _ = x.size()

        cls_tokens = self.cls_token.expand(
            batch_size, -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks

        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            if x.shape[1] != self.pos_embed.shape[1]:
                x = x + self.interpolate_pos_encoding(x, w, h)
            else:
                x = x + self.pos_embed

        for blk in self.blocks:
            x = blk(x)

        t = x[:, 1:, :]
        x = self.fc_norm(t)

        x = self.task_layer(x)
        return x, memo


@VQEncoderRegistry.register_()
class VQKDEncoder(VQKDMixin, BaseEncoder):
    """Vision Transformer with support for patch or hybrid CNN input stage."""

    def __init__(
        self,
        *args,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        out_chans: int = 32,
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            out_chans=out_chans,
            **kwargs,
        )

    @property
    def out_channels(self) -> int:
        return self.out_chans

    def forward(
        self,
        x: torch.Tensor,
        memo: Memo,
    ) -> tuple[torch.Tensor, Memo]:
        _, _, w, h = x.shape
        x, memo = super().forward(x, memo)
        assert h % self.patch_size == 0 and w % self.patch_size == 0
        h_prime = h // self.patch_size
        w_prime = w // self.patch_size
        x = einops.rearrange(
            x,
            'b (h w) c -> b c h w',
            h=h_prime,
            w=w_prime,
        )
        x = x.contiguous()
        return x, memo


@VQDecoderRegistry.register_()
class VQKDDecoder(VQKDMixin, BaseDecoder):
    """Vision Transformer with support for patch or hybrid CNN input stage."""

    def __init__(
        self,
        *args,
        img_size: int = 14,
        patch_size: int = 1,
        in_chans: int = 32,
        out_chans: int = 512,
        out_patch_size: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            out_chans=out_chans,
            out_patch_size=out_patch_size,
            **kwargs,
        )

    @property
    def in_channels(self) -> int:
        return self.in_chans

    @property
    def last_parameter(self) -> nn.Parameter:
        linear = self.task_layer[-1]
        assert isinstance(linear, nn.Linear)
        return cast(nn.Parameter, linear.weight)

    def forward(
        self,
        x: torch.Tensor,
        memo: Memo,
    ) -> tuple[torch.Tensor, Memo]:
        _, _, w, h = x.shape
        x, memo = super().forward(x, memo)
        x = einops.rearrange(
            x,
            'b (h w) (p q c) -> b c (h p) (w q)',
            h=h,
            w=w,
            p=self.out_patch_size,
            q=self.out_patch_size,
            c=self.out_chans,
        )
        return x, memo
