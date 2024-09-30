__all__ = [
    'VQITModelRegistry',
    'VQITRunnerRegistry',
]

from vq import VQModelRegistry, VQRunnerRegistry

from ..registries import VQITRegistry


class VQITModelRegistry(VQITRegistry, VQModelRegistry):
    pass


class VQITRunnerRegistry(VQITRegistry, VQRunnerRegistry):
    pass
