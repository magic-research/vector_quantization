__all__ = [
    'VQITQuantizerCallbackRegistry',
    'VQITQuantizerLossRegistry',
]

from ..registries import VQITLossRegistry, VQITQuantizerRegistry


class VQITQuantizerCallbackRegistry(VQITQuantizerRegistry):
    pass


class VQITQuantizerLossRegistry(VQITQuantizerRegistry, VQITLossRegistry):
    pass
