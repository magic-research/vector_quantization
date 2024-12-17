import argparse
import pathlib
from typing import Iterator

from todd.configs import PyConfig
from todd.datasets.access_layers import PILAccessLayer
from todd.patches.py_ import DictAction
from todd.patches.torch import get_world_size

from vq import VQRunnerRegistry
from vq.datasets import Dataset
from vq.utils import log

from ...runners.base import BaseValidator

# TODO


class SingletonAccessLayer(PILAccessLayer):

    def __init__(self, *args, singleton: str | pathlib.Path, **kwargs) -> None:
        if isinstance(singleton, str):
            singleton = pathlib.Path(singleton)
        singleton = singleton.absolute()
        super().__init__(
            *args,
            data_root=singleton.parent,
            suffix=singleton.suffix.removeprefix('.'),
            **kwargs,
        )

        self._singleton = singleton

    def _files(self) -> Iterator[pathlib.Path]:
        yield self._singleton


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('name')
    parser.add_argument('config', type=pathlib.Path)
    parser.add_argument('singleton')
    parser.add_argument('--config-options', action=DictAction, default=dict())
    parser.add_argument('--override', action=DictAction, default=dict())
    parser.add_argument('--load-model-from', required=True, nargs='+')
    parser.add_argument('--save')
    args = parser.parse_args()
    return args


def main() -> None:
    assert get_world_size() <= 1

    args = parse_args()

    config: PyConfig = PyConfig.load(args.config, **args.config_options)
    config = PyConfig(
        type=config.validator.type,
        strategy=config.validator.strategy,
        model=config.validator.model,
        callbacks=[],
        dataset=dict(
            type=Dataset.__name__,
            name=config.validator.dataset.name,
            num_categories=config.validator.dataset.num_categories,
            image_size=config.validator.dataset.image_size,
            access_layer=SingletonAccessLayer(singleton=args.singleton),
            transforms=config.validator.dataset.transforms,
        ),
        dataloader=dict(batch_size=1, num_workers=0),
        name=f'{args.name}_inference',
        save=args.save,
    )

    config.override(args.override)

    runner: BaseValidator = VQRunnerRegistry.build(config)
    log(runner, args, config)
    runner.strategy.load_model_from(args.load_model_from, strict=False)
    memo = runner.run()
    runner.logger.info("\n%s", memo['quantize']['quant'])  # TODO


if __name__ == '__main__':
    main()
