__all__ = [
    'VQSMTransformerRegistry',
    'VQSMSamplerRegistry',
]

from typing import TYPE_CHECKING, Any

import todd
from todd.bases.registries import Item

from ..registries import VQSMModelRegistry

if TYPE_CHECKING:
    from .samplers import BaseSampler


class VQSMTransformerRegistry(VQSMModelRegistry):
    pass


class VQSMSamplerRegistry(VQSMModelRegistry):

    @classmethod
    def _build(cls, item: Item, config: todd.Config) -> Any:
        config = config.copy()
        cfg = config.pop('cfg', None)
        sampler: 'BaseSampler' = todd.RegistryMeta._build(cls, item, config)
        if cfg is not None:
            from .samplers import (  # pylint: disable=import-outside-toplevel
                CFGSampler,
            )
            sampler = CFGSampler(sampler=sampler, alpha=cfg)
        return sampler
