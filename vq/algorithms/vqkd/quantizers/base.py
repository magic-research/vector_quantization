__all__ = [
    'VQKDQuantizer',
]

import todd

from vq.algorithms.vq import VectorQuantizer
from vq.tasks.image_tokenization.models import VQITQuantizerRegistry


@VQITQuantizerRegistry.register_()
class VQKDQuantizer(VectorQuantizer):

    def _init_weights(self, config: todd.Config) -> bool:
        return False
