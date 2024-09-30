__all__ = [
    'LAIONAestheticsDataset',
]

import random
from typing import cast

import todd
from PIL import Image
from todd.utils import retry

from ..registries import VQDatasetRegistry
from .base import T
from .split import SplitMixin


@VQDatasetRegistry.register_()
class LAIONAestheticsDataset(
    SplitMixin[str, Image.Image],
    todd.datasets.LAIONAestheticsDataset,
):

    def _rand_index(self) -> int:
        return random.randint(0, len(self) - 1)  # nosec B311

    @retry(10)
    def __getitem__(self, index: int, *, retry_times: int) -> T:
        if retry_times > 0:
            index = self._rand_index()
        item = super().__getitem__(  # type: ignore[safe-super]
            self._align_index(index),
        )
        image = item['image']
        item_ = cast(T, item)
        item_['original_image'] = image
        item_['image'] = self.encode(image)
        item_['category'] = 0
        return item_
