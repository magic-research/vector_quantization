__all__ = [
    'VQDiscriminatorLossRegistry',
    'VQGeneratorLossRegistry',
]

from vq.models import VQLossRegistry


class VQDiscriminatorLossRegistry(VQLossRegistry):
    pass


class VQGeneratorLossRegistry(VQLossRegistry):
    pass
