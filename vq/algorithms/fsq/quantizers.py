__all__ = [
    'FiniteScalarQuantizer',
]

from functools import cached_property
from typing import Iterable, cast

import einops
import torch
from todd.runners import Memo
from torch import nn

from vq.algorithms.sq import ScalarQuantizer
from vq.tasks.image_tokenization.models import VQITQuantizerRegistry
from vq.tasks.image_tokenization.models.quantizers.utils import ste
from vq.utils import get_memo


class BaseConverter(nn.Module):

    def __init__(
        self,
        *args,
        max_per_digit: Iterable[int],
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        max_per_digit = tuple(max_per_digit)

        cumprod = torch.tensor((1, ) + max_per_digit[:-1]).cumprod(0)
        self.register_buffer('_cumprod', cumprod, persistent=False)

        max_per_digit = torch.tensor(max_per_digit, dtype=torch.int)
        self.register_buffer(
            '_max_per_digit',
            max_per_digit,
            persistent=False,
        )

    def __len__(self) -> int:
        return cast(int, self.max_per_digit.prod().item())

    @cached_property
    def num_digits(self) -> int:
        return self.max_per_digit.numel()

    @property
    def max_(self) -> torch.Tensor:
        return self.max_per_digit - 1

    @property
    def max_per_digit(self) -> torch.Tensor:
        return self.get_buffer('_max_per_digit')

    @property
    def cumprod(self) -> torch.Tensor:
        return self.get_buffer('_cumprod')

    def from_decimal(self, x: torch.Tensor) -> torch.Tensor:
        x = einops.rearrange(x, '... -> ... 1')
        x = x // self.cumprod
        x = x % self.max_per_digit
        return x

    def to_decimal(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.cumprod
        x = x.sum(-1).to(torch.int)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.from_decimal(x)


@VQITQuantizerRegistry.register_()
class FiniteScalarQuantizer(ScalarQuantizer):

    def __init__(
        self,
        *args,
        eps: float = 1e-3,
        num_scalars_per_channel: tuple[int, ...],
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._eps = eps

        base_converter = BaseConverter(max_per_digit=num_scalars_per_channel)
        self._base_converter = base_converter

        quant = torch.arange(self.codebook_size)
        quant = self._base_converter(quant)
        embeddings = quant / (self._base_converter.max_per_digit // 2) - 1
        self.register_buffer('_embeddings', embeddings)

    @property
    def embedding_dim(self) -> int:
        # return self._embedding_dim
        return self._base_converter.num_digits

    @cached_property
    def codebook_size(self) -> int:
        return len(self._base_converter)

    @property
    def embeddings(self) -> torch.Tensor:
        return self.get_buffer('_embeddings')

    def _encode(
        self,
        x: torch.Tensor,
        memo: Memo,
    ) -> tuple[torch.Tensor, Memo]:
        # TODO: try if other implementation works
        max_ = self._base_converter.max_ * (1 - self._eps)
        odd = self._base_converter.max_ % 2

        # z = self._in_proj(x)
        z = torch.tanh(x + torch.atanh(odd / max_)) * max_ - odd
        z = z / 2
        z = ste(z.round(), z)

        half_max = self._base_converter.max_per_digit // 2
        memo['z'] = z / half_max
        z = z + half_max
        quant = self._base_converter.to_decimal(z)
        return quant, memo

    def _decode(
        self,
        quant: torch.Tensor,
        memo: Memo,
    ) -> tuple[torch.Tensor, Memo]:
        if 'z' in memo:
            z = memo['z']
        else:
            quant = self._base_converter(quant)
            z = quant / (self._base_converter.max_per_digit // 2) - 1
        # z = self._out_proj(z)
        return z, memo

    def decode(
        self,
        quant: torch.Tensor,
        memo: Memo,
    ) -> tuple[torch.Tensor, Memo]:
        encode_memo = get_memo(memo, 'encode')
        decode_memo = get_memo(memo, 'decode')
        if 'z' in encode_memo:
            decode_memo['z'] = encode_memo['z']
        return super().decode(quant, memo)
