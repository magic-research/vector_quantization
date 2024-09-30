__all__ = [
    'SAMed2DDataset',
]

from typing import cast

import todd
from PIL import Image

from ..registries import VQDatasetRegistry
from .base import T
from .split import SplitMixin


@VQDatasetRegistry.register_()
class SAMed2DDataset(
    SplitMixin[str, Image.Image],
    todd.datasets.SAMed2DDataset,
):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, shuffle=True, **kwargs)

    def __getitem__(self, index: int) -> T:  # type: ignore[override]
        item = super().__getitem__(  # type: ignore[safe-super]
            self._align_index(index),
        )
        image = item['image']
        item_ = cast(T, item)
        item_['original_image'] = image
        item_['image'] = self.encode(image)
        item_['category'] = 0
        return item_
