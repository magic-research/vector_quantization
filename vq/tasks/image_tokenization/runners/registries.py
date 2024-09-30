__all__ = [
    'VQITCallbackRegistry',
    'VQITMetricRegistry',
]

from vq.runners import VQCallbackRegistry, VQMetricRegistry

from ..registries import VQITRunnerRegistry


class VQITCallbackRegistry(VQITRunnerRegistry, VQCallbackRegistry):
    pass


class VQITMetricRegistry(VQITRunnerRegistry, VQMetricRegistry):
    pass
