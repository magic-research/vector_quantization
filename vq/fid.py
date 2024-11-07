import argparse

from todd.configs import PyConfig
from todd.patches.py_ import DictAction
from todd.registries import RunnerRegistry
from torch import nn

from .runners import BaseValidator
from .utils import log


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('dataset')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--strategy', default='cuda')
    parser.add_argument('--override', action=DictAction, default=dict())
    parser.add_argument('--autocast', action='store_true')
    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()
    config: PyConfig = PyConfig.load(
        'configs/fid/interface.py',
        dataset=args.dataset,
        strategy=args.strategy,
    )
    config = config.trainer if args.train else config.validator
    config.override(args.override)

    name = args.dataset
    if args.train:
        name += '_train'
    name += f'_{args.strategy}'

    runner: BaseValidator[nn.Module] = RunnerRegistry.build(
        config,
        name=f'fid/{name}',
        autocast=args.autocast,
    )
    log(runner, args, config)
    runner.run()


if __name__ == '__main__':
    main()
