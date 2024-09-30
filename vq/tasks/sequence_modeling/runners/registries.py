__all__ = [
    'VQSMMetricRegistry',
]

from vq.runners import VQMetricRegistry

from ..registries import VQSMRunnerRegistry


class VQSMMetricRegistry(VQSMRunnerRegistry, VQMetricRegistry):
    pass
