__all__ = [
    'Dataset',
]

import todd
import torch
import torchvision.transforms.functional as F
from PIL import Image

from ..registries import VQDatasetRegistry
from .base import BaseMixin, T


@VQDatasetRegistry.register_()
class Dataset(BaseMixin[str, Image.Image], todd.datasets.PILDataset[T]):

    def _transform(self, image: Image.Image) -> torch.Tensor:
        if self._transforms is None:
            return F.pil_to_tensor(image)
        return self._transforms(image)

    def __getitem__(self, index: int) -> T:
        key, image = self._access(index)
        tensor = self._transform(image)
        encoded_tensor = self.encode(tensor)
        return T(
            id_=key,
            original_image=tensor,
            image=encoded_tensor,
            category=0,
        )
