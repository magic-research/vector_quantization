__all__ = [
    'CVQVAECallback',
]

import einops
import todd
import torch
import torch.distributed
from todd.bases.registries import Item
from todd.runners import Memo

from vq.algorithms.vq.callbacks import UpdateMixin
from vq.algorithms.vq.utils import QuantStatistics
from vq.tasks.image_tokenization.models.quantizers import (
    VQITQuantizerCallbackRegistry,
)
from vq.tasks.image_tokenization.models.quantizers.callbacks import (
    BaseCallback,
)

from .anchors import BaseAnchor
from .registries import AnchorRegistry


@VQITQuantizerCallbackRegistry.register_()
class CVQVAECallback(UpdateMixin, BaseCallback):

    def __init__(
        self,
        *args,
        anchor: BaseAnchor,
        eps: float = 1e-3,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._anchor = anchor
        self._eps = eps

    @classmethod
    def anchor_build_pre_hook(
        cls,
        config: todd.Config,
        registry: todd.RegistryMeta,
        item: Item,
    ) -> todd.Config:
        config.anchor = AnchorRegistry.build_or_return(config.anchor)
        return config

    @classmethod
    def build_pre_hook(
        cls,
        config: todd.Config,
        registry: todd.RegistryMeta,
        item: Item,
    ) -> todd.Config:
        config = super().build_pre_hook(config, registry, item)
        config = cls.anchor_build_pre_hook(config, registry, item)
        return config

    def before_init_weights(self, config: todd.Config) -> None:
        super().before_init_weights(config)
        if not self.quantizer.training:
            return

        p = torch.zeros(self.quantizer.codebook_size)
        self._update_probability(p)

    @property
    def probability(self) -> torch.Tensor:
        return self.quantizer.get_buffer('_probability')

    def _update_probability(self, value: torch.Tensor) -> None:
        self.quantizer.register_buffer('_probability', value)

    def after_encode(
        self,
        x: torch.Tensor,
        quant: torch.Tensor,
        memo: Memo,
    ) -> torch.Tensor:
        quant = super().after_encode(x, quant, memo)
        if not self.quantizer.training:
            return quant

        e = self.quantizer.embeddings
        d = memo['encode']['distance']

        quant_statistics = QuantStatistics(
            quant=quant,
            codebook_size=self.quantizer.codebook_size,
            sync=True,
        )
        frequency = quant_statistics.frequency()
        p = self._ema(self.probability, frequency)
        self._update_probability(p)

        anchors, memo = self._anchor(x, e, d, quant, p, memo=memo)
        decay = 1 - torch.exp(
            -einops.rearrange(p, 'e -> e 1') * self.quantizer.codebook_size
            * 10 / (1 - self._ema.decay) - self._eps,
        )
        e = todd.utils.ema(e, anchors, decay)
        self._update_embedding(e)

        return quant
