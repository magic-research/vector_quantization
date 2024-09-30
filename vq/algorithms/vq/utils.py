__all__ = [
    'QuantStatistics',
]

import functools
from typing_extensions import Self

import torch
import torch.distributed
from todd.patches.torch import get_world_size


class QuantStatistics:

    def __init__(
        self,
        *args,
        quant: torch.Tensor,
        codebook_size: int,
        sync: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._quant = quant
        self._codebook_size = codebook_size
        self._sync = sync and get_world_size() > 1

    @staticmethod
    def sync_decorator(func):

        @functools.wraps(func)
        def wrapper(self: Self, *args, **kwargs) -> torch.Tensor:
            tensor = func(self, *args, **kwargs)
            if self._sync:
                torch.distributed.all_reduce(tensor)
            return tensor

        return wrapper

    @sync_decorator
    def bin_count(self) -> torch.Tensor:
        return self._quant.bincount(minlength=self._codebook_size)

    @sync_decorator
    def num_elements(self) -> torch.Tensor:
        return self._quant.new_tensor(self._quant.numel())

    def frequency(self) -> torch.Tensor:
        bin_count = self.bin_count()
        numel = self.num_elements()
        frequency = bin_count / numel
        return frequency
