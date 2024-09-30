__all__ = [
    'Store',
]

from todd.utils import StoreMeta


class Store(metaclass=StoreMeta):
    DEBUG: bool
    PRETRAINED: str
