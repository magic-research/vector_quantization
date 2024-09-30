__all__ = [
    'VQIRModelRegistry',
    'VQIRLossRegistry',
]

from vq import VQModelRegistry
from vq.models import VQLossRegistry

from ..registries import VQIRRegistry


class VQIRModelRegistry(VQIRRegistry, VQModelRegistry):
    pass


class VQIRLossRegistry(VQIRModelRegistry, VQLossRegistry):
    pass
