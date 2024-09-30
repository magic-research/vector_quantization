__all__ = [
    'VQGANQuantizer',
]

import todd

from vq.algorithms.vq import VectorQuantizer
from vq.tasks.image_tokenization.models import VQITQuantizerRegistry


@VQITQuantizerRegistry.register_()
class VQGANQuantizer(VectorQuantizer):

    def _init_weights(self, config: todd.Config) -> bool:
        if config == todd.Config(type='vqgan'):
            config = todd.Config(
                type='uniform_',
                a=-1.0 / self.codebook_size,
                b=1.0 / self.codebook_size,
            )
        return super()._init_weights(config)
