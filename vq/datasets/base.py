__all__ = [
    'Batch',
    'BaseMixin',
]

from abc import ABC
from typing import TypedDict, TypeVar

import torch
from todd.datasets import BaseDataset

KT = TypeVar('KT')
VT = TypeVar('VT')


class T(TypedDict):
    id_: str
    original_image: torch.Tensor
    image: torch.Tensor
    category: int


class Batch(TypedDict):
    id_: list[str]
    original_image: torch.Tensor
    image: torch.Tensor
    category: torch.Tensor


class BaseMixin(BaseDataset[T, KT, VT], ABC):

    def __init__(
        self,
        *args,
        name: str,
        num_categories: int,
        image_size: int,
        fid_path: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._name = name
        self._num_categories = num_categories
        self._image_size = image_size
        self._fid_path = fid_path

    @property
    def name(self) -> str:
        return self._name

    @property
    def num_categories(self) -> int:
        return self._num_categories

    @property
    def image_size(self) -> int:
        return self._image_size

    @property
    def fid_path(self) -> str:
        if self._fid_path is None:
            return f'pretrained/fid/{self._name}.pth'
        return self._fid_path

    @classmethod
    def encode(cls, images: torch.Tensor) -> torch.Tensor:
        return images / 127.5 - 1.0

    @classmethod
    def decode(cls, images: torch.Tensor) -> torch.Tensor:
        images = (images + 1) * 127.5
        images = images.clamp(0, 255).to(torch.uint8)
        return images
