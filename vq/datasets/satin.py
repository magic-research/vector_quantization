__all__ = [
    'SATINDataset',
]

from typing import Any

import todd

from ..registries import VQDatasetRegistry
from .base import T
from .split import SplitMixin


@VQDatasetRegistry.register_()
class SATINDataset(
    SplitMixin[int, dict[str, Any]],
    todd.datasets.SATINDataset,
):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, shuffle=True, **kwargs)

    def _access(  # type: ignore[override]
        self,
        index: int,
    ) -> tuple[int, dict[str, Any]]:
        return todd.datasets.SATINDataset._access(  # type: ignore[return-value] # noqa: E501 pylint: disable=line-too-long
            self,  # type: ignore[arg-type]
            index,
        )

    def __getitem__(self, index: int) -> T:  # type: ignore[override]
        item = super().__getitem__(  # type: ignore[safe-super]
            self._align_index(index),
        )
        id_ = item['id_']
        image = item['image']
        return T(
            id_=str(id_),
            image=self.encode(image),
            original_image=image,
            category=0,
        )
