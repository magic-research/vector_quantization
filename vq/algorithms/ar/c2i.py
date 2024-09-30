__all__ = [
    'ARC2I',
]

import todd.tasks.large_multimodal_model as lmm

from vq import VQModelRegistry
from vq.tasks.sequence_modeling.models import C2I

from .x2i import X2IMixin


@VQModelRegistry.register_()
class ARC2I(X2IMixin[lmm.C2IEnum], C2I):
    pass
