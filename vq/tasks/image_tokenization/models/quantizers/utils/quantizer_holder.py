__all__ = [
    'QuantizerHolderMixin',
]

from typing import TYPE_CHECKING

import todd

from ..base import BaseQuantizer

if TYPE_CHECKING:
    from vq.algorithms.vq import VectorQuantizer


class QuantizerHolderMixin(todd.utils.HolderMixin[BaseQuantizer]):

    @property
    def quantizer(self) -> BaseQuantizer:
        return self._instance

    @property
    def vector_quantizer(self) -> 'VectorQuantizer':
        from vq.algorithms.vq import (  # noqa: E501 pylint: disable=import-outside-toplevel
            VectorQuantizer,
        )
        assert isinstance(self.quantizer, VectorQuantizer)
        return self.quantizer
