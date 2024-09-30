__all__ = [
    'Tokens',
    'TokenizeCallback',
]

import pathlib
from typing import TypedDict, TypeVar

import einops
import torch
from todd.patches.torch import get_rank
from todd.runners import Memo
from todd.runners.callbacks import BaseCallback
from torch import nn

from vq.datasets import Batch

from .registries import VQITCallbackRegistry

T = TypeVar('T', bound=nn.Module)


class Tokens(TypedDict):
    id_: list[str]
    category: torch.Tensor
    tokens: torch.Tensor


@VQITCallbackRegistry.register_()
class TokenizeCallback(BaseCallback[T]):

    @property
    def token_dir(self) -> pathlib.Path:
        return self.runner.work_dir / 'tokens'

    def bind(self, *args, **kwargs) -> None:
        super().bind(*args, **kwargs)
        self.token_dir.mkdir(parents=True, exist_ok=True)

    def after_run_iter(self, batch: Batch, memo: Memo) -> None:
        super().after_run_iter(batch, memo)
        quantizer_memo = memo['quantizer']
        quant = quantizer_memo['quant']
        b, _, h, w = quantizer_memo['x_shape']
        tokens = einops.rearrange(quant, '(b h w) -> b h w', b=b, h=h, w=w)
        torch.save(
            Tokens(
                id_=batch['id_'],
                category=batch['category'],
                tokens=tokens,
            ),
            self.token_dir / f'{self.runner.iter_}_{get_rank()}.pth',
        )
