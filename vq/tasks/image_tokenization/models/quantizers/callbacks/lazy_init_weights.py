__all__ = [
    'LazyInitWeightsMixin',
]

from abc import abstractmethod

import todd
import torch
from todd.runners import Memo

from ..base import BaseQuantizer
from .base import BaseCallback


class LazyInitWeightsMixin(BaseCallback):

    @abstractmethod
    def lazy_init_weights(
        self,
        config: todd.Config,
        x: torch.Tensor,
        memo: Memo,
    ) -> None:
        pass

    def before_init_weights(self, config: todd.Config) -> None:
        super().before_init_weights(config)
        lazy_init_weights = config.pop('lazy_init_weights', todd.Config())

        def forward_pre_hook(
            module: BaseQuantizer,
            args: tuple[torch.Tensor, Memo],
        ) -> None:
            x, memo = args
            self.lazy_init_weights(lazy_init_weights, x, memo)
            handle.remove()

        handle = self.quantizer.register_forward_pre_hook(forward_pre_hook)
