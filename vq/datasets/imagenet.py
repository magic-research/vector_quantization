__all__ = [
    'ImageNetDataset',
]

from typing import cast

import todd
from PIL import Image

from ..registries import VQDatasetRegistry
from .base import BaseMixin, T


@VQDatasetRegistry.register_()
class ImageNetDataset(
    BaseMixin[str, Image.Image],
    todd.datasets.ImageNetDataset,
):

    def __getitem__(self, index: int) -> T:
        item = super().__getitem__(index)
        image = item['image']
        item = cast(T, item)
        item['original_image'] = image
        item['image'] = self.encode(image)
        return item
