__all__ = [
    'VQSMModelRegistry',
    'VQSMRunnerRegistry',
]

from vq import VQModelRegistry, VQRunnerRegistry

from ..registries import VQSMRegistry


class VQSMModelRegistry(VQSMRegistry, VQModelRegistry):
    pass


class VQSMRunnerRegistry(VQSMRegistry, VQRunnerRegistry):
    pass
