import argparse
import importlib
import pathlib
from typing import TypeVar, cast

import numpy as np
import todd
import torch
import torchvision.transforms as tf
from todd.bases.registries import Item
from todd.configs import PyConfig
from todd.patches.py_ import DictAction
from todd.runners import Memo
from todd.runners.callbacks import BaseCallback
from torch import nn

from vq.datasets import Batch
from vq.registries import VQRunnerRegistry
from vq.tasks.image_tokenization import VQITRunnerRegistry
from vq.tasks.image_tokenization.runners import Tokenizer as Tokenizer_
from vq.tasks.image_tokenization.runners import VQITCallbackRegistry
from vq.utils import log

T = TypeVar('T', bound=nn.Module)


@VQITRunnerRegistry.register_(force=True)
class Tokenizer(Tokenizer_[T]):

    @classmethod
    def dataset_build_pre_hook(
        cls,
        config: todd.Config,
        registry: todd.RegistryMeta,
        item: Item,
    ) -> todd.Config:
        image_size = config.pop('image_size')
        config.dataset.image_size = image_size

        pil_to_tensor = tf.PILToTensor()
        config.dataset.transforms = [
            dict(type='Resize', size=int(image_size * 1.1)),
            dict(type='CenterCrop', size=int(image_size * 1.1)),
            dict(type='TenCrop', size=image_size),
            dict(
                type='Lambda',
                lambd=lambda crops: torch.
                stack([pil_to_tensor(crop) for crop in crops]),
            ),
        ]
        return super().dataset_build_pre_hook(config, registry, item)

    @classmethod
    def dataloader_build_pre_hook(
        cls,
        config: todd.Config,
        registry: todd.RegistryMeta,
        item: Item,
    ) -> todd.Config:
        dataloader: todd.Config = config.dataloader
        dataloader.update(batch_size=1, batch_size_in_total=False)
        return super().dataloader_build_pre_hook(config, registry, item)


@VQITCallbackRegistry.register_(force=True)
class TokenizeCallback(BaseCallback[T]):
    runner: Tokenizer[T]

    @property
    def token_dir(self) -> pathlib.Path:
        return self.runner.work_dir / 'llamagen_tokens'

    @property
    def code_dir(self) -> pathlib.Path:
        image_size = self.runner.dataset.image_size
        return self.token_dir / f'imagenet{image_size}_codes'

    @property
    def label_dir(self) -> pathlib.Path:
        image_size = self.runner.dataset.image_size
        return self.token_dir / f'imagenet{image_size}_labels'

    def bind(self, *args, **kwargs) -> None:
        super().bind(*args, **kwargs)
        self.code_dir.mkdir(parents=True, exist_ok=True)
        self.label_dir.mkdir(parents=True, exist_ok=True)

    def before_run_iter(self, batch: Batch, memo: Memo) -> None:
        batch['original_image'] = batch['original_image'].squeeze(0)
        batch['image'] = batch['image'].squeeze(0)
        return super().before_run_iter(batch, memo)

    def after_run_iter(self, batch: Batch, memo: Memo) -> None:
        world_size = todd.patches.torch.get_world_size()
        rank = todd.patches.torch.get_rank()
        i = (self.runner.iter_ - 1) * world_size + rank

        quant: torch.Tensor = memo['quantizer']['quant']
        quant = quant.reshape((1, 10, -1))
        np.save(self.code_dir / f'{i}.npy', quant.cpu().numpy())

        categories = batch['category']
        np.save(self.label_dir / f'{i}.npy', categories.numpy())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('name')
    parser.add_argument('config', type=pathlib.Path)
    parser.add_argument('--config-options', action=DictAction, default=dict())
    parser.add_argument('--image-size', type=int, default=256)
    parser.add_argument('--override', action=DictAction, default=dict())
    parser.add_argument('--autocast', action='store_true')
    parser.add_argument('--load-model-from', required=True, nargs='+')
    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()
    config = PyConfig.load(args.config, **args.config_options)
    cast(PyConfig, config.validator).update(
        type='VQITRunnerRegistry.Tokenizer',
        image_size=args.image_size,
    )
    config.validator.dataset = config.trainer.dataset  # do not use update
    config.override(args.override)

    for custom_import in config.get('custom_imports', []):
        importlib.import_module(custom_import)

    runner: Tokenizer[nn.Module] = VQRunnerRegistry.build(
        config.validator,
        name=f'llamagen/{args.name}',
        autocast=args.autocast,
    )
    log(runner, args, config)
    runner.strategy.load_model_from(args.load_model_from, strict=False)
    runner.run()


if __name__ == '__main__':
    main()
