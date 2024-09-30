__all__ = [
    'SplitMixin',
]

from abc import ABC
from typing import TypeVar

import todd

from ..registries import VQDatasetRegistry
from .base import BaseMixin

KT = TypeVar('KT')
VT = TypeVar('VT')


@VQDatasetRegistry.register_()
class SplitMixin(BaseMixin[KT, VT], ABC):

    def __init__(
        self,
        *args,
        train: bool,
        num_val_samples: int = 25_000,
        shuffle: bool = False,
        **kwargs,
    ) -> None:
        self._train = train
        self._num_val_samples = num_val_samples
        self._shuffle = shuffle
        super().__init__(*args, **kwargs)

    def __len__(self) -> int:
        if todd.Store.DRY_RUN:
            return super().__len__()
        if self._train:
            return super().__len__() - self._num_val_samples
        return self._num_val_samples

    def _align_index(self, index: int) -> int:
        if todd.Store.DRY_RUN or not self._shuffle:
            return index if self._train else super().__len__() - 1 - index

        chunk_size = super().__len__() // self._num_val_samples
        assert chunk_size > 1

        if self._train:
            chunk_id = index // (chunk_size - 1)
            chunk_id = min(chunk_id, self._num_val_samples - 1)
            return index + 1 + chunk_id
        return index * chunk_size
