__all__ = [
    'ClusterEncoder',
]

import todd
import torch
from todd.bases.registries import BuildPreHookMixin, Item
from todd.runners import Memo

from vq.algorithms.vqkd import VQTeacherRegistry
from vq.algorithms.vqkd.teachers import BaseTeacher
from vq.models import BaseEncoder, VQEncoderRegistry


@VQEncoderRegistry.register_()
class ClusterEncoder(BuildPreHookMixin, BaseEncoder):

    def __init__(self, *args, teacher: BaseTeacher, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._teacher = teacher

    @classmethod
    def teacher_build_pre_hook(
        cls,
        config: todd.Config,
        registry: todd.RegistryMeta,
        item: Item,
    ) -> todd.Config:
        config.teacher = VQTeacherRegistry.build_or_return(config.teacher)
        return config

    @classmethod
    def build_pre_hook(
        cls,
        config: todd.Config,
        registry: todd.RegistryMeta,
        item: Item,
    ) -> todd.Config:
        config = super().build_pre_hook(config, registry, item)
        config = cls.teacher_build_pre_hook(config, registry, item)
        return config

    @property
    def out_channels(self) -> int:
        return self._teacher.out_channels

    def forward(
        self,
        image: torch.Tensor,
        memo: Memo,
    ) -> tuple[torch.Tensor, Memo]:
        x = self._teacher(memo['original_image'], return_2d=True)
        return x, memo
