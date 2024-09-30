__all__ = [
    'VQEncoderRegistry',
    'VQDecoderRegistry',
    'VQLossRegistry',
]

from todd.models import LossRegistry

from ..registries import VQModelRegistry, VQRegistry


class VQEncoderRegistry(VQModelRegistry):
    pass


class VQDecoderRegistry(VQModelRegistry):
    pass


class VQLossRegistry(VQRegistry, LossRegistry):
    pass
