__all__ = [
    'BaseMixin',
]

import enum
from abc import abstractmethod
from typing import Any, TypeVar

import torch
from todd.runners import Memo

from vq.tasks.sequence_modeling.models import BaseModel as BaseSMModel

T = TypeVar('T', bound=enum.Enum)


class BaseMixin(BaseSMModel[T]):

    @abstractmethod
    def sample(
        self,
        logits: torch.Tensor,
        memo: Memo,
    ) -> tuple[Any, Memo]:
        pass
