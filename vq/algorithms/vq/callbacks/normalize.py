__all__ = [
    'NormalizeCallback',
]

import torch
import torch.nn.functional as F
from todd.runners import Memo

from vq.tasks.image_tokenization.models.quantizers import (
    VQITQuantizerCallbackRegistry,
)
from vq.tasks.image_tokenization.models.quantizers.callbacks import (
    BaseCallback,
)

from .update import UpdateMixin


@VQITQuantizerCallbackRegistry.register_()
class NormalizeCallback(UpdateMixin, BaseCallback):

    def before_encode(self, x: torch.Tensor, memo: Memo) -> torch.Tensor:
        x = super().before_encode(x, memo)
        x = F.normalize(x)

        e = self.vector_quantizer.embedding.weight
        e = F.normalize(e)
        self._update_embedding(e)
        return x
