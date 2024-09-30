import argparse
import importlib
import pathlib

from todd.configs import PyConfig
from todd.patches.py import DictAction
from torch import nn

from vq.registries import VQRunnerRegistry
from vq.utils import log

from .runners import Tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('name')
    parser.add_argument('config', type=pathlib.Path)
    parser.add_argument('--config-options', action=DictAction, default=dict())
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--override', action=DictAction, default=dict())
    parser.add_argument('--autocast', action='store_true')
    parser.add_argument('--load-model-from', required=True, nargs='+')
    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()
    config = PyConfig.load(args.config, **args.config_options)
    config.validator.type = 'VQITRunnerRegistry.Tokenizer'
    if args.train:
        config.validator.dataset = config.trainer.dataset
    config.override(args.override)

    for custom_import in config.get('custom_imports', []):
        importlib.import_module(custom_import)

    name = f'{args.name}_tokenize'
    if args.train:
        name += '_train'

    runner: Tokenizer[nn.Module] = VQRunnerRegistry.build(
        config.validator,
        name=name,
        autocast=args.autocast,
    )
    log(runner, args, config)
    runner.strategy.load_model_from(args.load_model_from, strict=False)
    runner.run()


if __name__ == '__main__':
    main()
