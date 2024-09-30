__all__ = [
    'ScalarQuantizer',
]

from vq.tasks.image_tokenization.models import VQITQuantizerRegistry
from vq.tasks.image_tokenization.models.quantizers import BaseQuantizer


@VQITQuantizerRegistry.register_()
class ScalarQuantizer(BaseQuantizer):
    pass
