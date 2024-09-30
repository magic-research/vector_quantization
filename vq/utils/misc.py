__all__ = [
    'get_memo',
    'device',
    'log',
    'load',
    'todd_version',
]

import argparse
import functools
import importlib
import pathlib
import sys
from typing import TYPE_CHECKING, Iterable, Literal, TypeVar, cast

import todd
import torch
from todd.configs import PyConfig
from todd.patches.torch import get_rank, load_state_dict, load_state_dict_
from todd.registries import ModelRegistry
from todd.runners import Memo
from torch import nn

if TYPE_CHECKING:
    from ..runners import BaseMixin as BaseRunnerMixin

T = TypeVar('T', bound=nn.Module)


def get_memo(memo: Memo, key: str) -> Memo:
    memo_: Memo
    if key in memo:
        memo_ = memo[key]
        assert isinstance(memo_, dict)
    else:
        memo_ = dict()
        memo[key] = memo_
    return memo_


@functools.lru_cache(1)
def device() -> str:
    return 'cuda' if todd.Store.cuda else 'cpu'


def log(
    runner: 'BaseRunnerMixin[T]',
    args: argparse.Namespace,
    config: PyConfig,
) -> None:
    if get_rank() != 0:
        return

    runner.logger.info("Command\n" + ' '.join(sys.argv))
    runner.logger.info(f"Args\n{vars(args)}")
    runner.logger.info(f"Config\n{config.dumps()}")

    if 'config' in args:
        config_name = cast(pathlib.Path, args.config).name
        PyConfig(config).dump(runner.work_dir / config_name)


def load(
    *args,
    config: str,
    runner_type: Literal['trainer', 'validator'] = 'validator',
    state_dicts: Iterable[torch.serialization.FILE_LIKE],
    **kwargs,
) -> nn.Module:
    config_ = PyConfig.load(config)
    for custom_import in config_.get('custom_imports', []):
        importlib.import_module(custom_import)
    config_ = config_[runner_type].model
    model: nn.Module = ModelRegistry.build(config_)

    state_dict = load_state_dict_(state_dicts)  # type: ignore[arg-type]
    load_state_dict(model, state_dict, *args, **kwargs)
    return model


@todd.utils.EnvRegistry.register_(force=True)
def todd_version(verbose: bool = False) -> str:
    with open('.todd_version') as f:
        commit_id = f.read().strip()
    version = f'{todd.utils.todd_version()}+{commit_id}'
    return version
