__all__ = [
    'build_module_list',
    'build_module_dict',
    'build_sequential',
]

from typing import Iterable

import todd
from todd.patches.torch import ModuleDict, ModuleList, Sequential


def build_module_list(
    registry: todd.RegistryMeta,
    config: Iterable[todd.Config | None],
    **kwargs,
) -> ModuleList:
    module_list = [
        registry.build_or_return(c, **kwargs) for c in config if c is not None
    ]
    return ModuleList(module_list)


def build_module_dict(
    registry: todd.RegistryMeta,
    config: todd.Config,
    **kwargs,
) -> ModuleDict:
    module_dict = {
        k: registry.build_or_return(v, **kwargs)
        for k, v in config.items()
        if v is not None
    }
    return ModuleDict(module_dict)


def build_sequential(
    registry: todd.RegistryMeta,
    config: Iterable[todd.Config | None],
    **kwargs,
) -> Sequential:
    sequential = [
        registry.build_or_return(c, **kwargs) for c in config if c is not None
    ]
    return Sequential(*sequential)
