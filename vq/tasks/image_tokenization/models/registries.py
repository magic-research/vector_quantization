__all__ = [
    'VQITConnectorRegistry',
    'VQITLossRegistry',
    'VQITQuantizerRegistry',
]

from vq.models import VQLossRegistry

from ..registries import VQITModelRegistry


class VQITConnectorRegistry(VQITModelRegistry):
    pass


class VQITLossRegistry(VQITModelRegistry, VQLossRegistry):
    pass


class VQITQuantizerRegistry(VQITModelRegistry):
    pass
