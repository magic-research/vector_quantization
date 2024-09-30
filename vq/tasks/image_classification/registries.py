__all__ = [
    'VQICModelRegistry',
]

from vq import VQModelRegistry

from ..registries import VQICRegistry


class VQICModelRegistry(VQICRegistry, VQModelRegistry):
    pass
