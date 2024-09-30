__all__ = [
    'BaseCallback',
]

import todd
import torch
from todd.runners import Memo

from ..utils import QuantizerHolderMixin


class BaseCallback(QuantizerHolderMixin):

    def before_init_weights(self, config: todd.Config) -> None:
        pass

    def after_init_weights(self, config: todd.Config, recursive: bool) -> bool:
        return recursive

    def before_encode(self, x: torch.Tensor, memo: Memo) -> torch.Tensor:
        return x

    def after_encode(
        self,
        x: torch.Tensor,
        quant: torch.Tensor,
        memo: Memo,
    ) -> torch.Tensor:
        return quant

    def before_decode(self, quant: torch.Tensor, memo: Memo) -> torch.Tensor:
        return quant

    def after_decode(self, z: torch.Tensor, memo: Memo) -> torch.Tensor:
        return z

    def before_loss(
        self,
        z: torch.Tensor,
        x: torch.Tensor,
        memo: Memo,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return z, x

    def after_loss(self, loss: torch.Tensor, memo: Memo) -> torch.Tensor:
        return loss
