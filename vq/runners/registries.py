__all__ = [
    'VQCallbackRegistry',
    'VQMetricRegistry',
]

from todd.runners import CallbackRegistry, MetricRegistry

from ..registries import VQRegistry


class VQCallbackRegistry(VQRegistry, CallbackRegistry):
    pass


class VQMetricRegistry(VQRegistry, MetricRegistry):
    pass
