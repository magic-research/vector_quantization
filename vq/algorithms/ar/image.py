__all__ = [
    'ImageMixin',
]

import enum
from typing import TypeVar

from vq.tasks.sequence_modeling.models import ImageModel

from .base import BaseMixin

T = TypeVar('T', bound=enum.Enum)


class ImageMixin(BaseMixin[T], ImageModel[T]):
    pass
