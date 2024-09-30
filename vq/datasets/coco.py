__all__ = [
    'COCODataset',
]

from typing import cast

import todd
from PIL import Image

from ..registries import VQDatasetRegistry
from .base import BaseMixin, T


@VQDatasetRegistry.register_()
class COCODataset(BaseMixin[str, Image.Image], todd.datasets.COCODataset):

    def __getitem__(self, index: int) -> T:  # type: ignore[override]
        item = super().__getitem__(index)
        image = item['image']
        item_ = cast(T, item)
        item_['original_image'] = image
        item_['image'] = self.encode(image)
        item_['category'] = 0
        return item_
