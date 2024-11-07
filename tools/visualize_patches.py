import argparse
import importlib
import pathlib
from collections import defaultdict
from typing import TypeVar

import cv2
import einops
import todd
import torch
import torchvision
import tqdm
from todd.configs import PyConfig
from todd.datasets.access_layers import PthAccessLayer
from todd.patches.pil import convert_rgb
from todd.patches.py_ import DictAction
from todd.registries import DatasetRegistry
from torch import nn

from vq.datasets import BaseMixin as BaseDatasetMixin
from vq.tasks.image_tokenization.runners import Tokens

T = TypeVar('T', bound=nn.Module)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('config', type=pathlib.Path)
    parser.add_argument('path')
    parser.add_argument('--config-options', action=DictAction, default=dict())
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--override', action=DictAction, default=dict())
    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()
    config = PyConfig.load(args.config, **args.config_options)
    config.override(args.override)

    for custom_import in config.get('custom_imports', []):
        importlib.import_module(custom_import)

    out = pathlib.Path(args.path).parent / 'visualize_patches'
    out.mkdir(parents=True, exist_ok=True)

    token2patches_path = out / 'token2patches.pth'
    if token2patches_path.exists():
        todd.logger.info("Loading token2patches")
        token2patches = torch.load(token2patches_path)
        todd.logger.info("Loading token2patches done")
    else:

        runner_config = config.trainer if args.train else config.validator
        runner_config.dataset.image_size = 512
        runner_config.dataset.transforms[0].size = 512
        dataset: BaseDatasetMixin = DatasetRegistry.build_or_return(
            runner_config.dataset,
        )
        transforms = dataset.transforms
        assert transforms is not None

        token2patches = defaultdict(list)

        pth_access_layer: PthAccessLayer[Tokens] = PthAccessLayer(
            data_root=args.path,
            task_name='',
        )
        for data in tqdm.tqdm(pth_access_layer.values(), "Loading data"):
            tokens = data['tokens']
            _, ny, nx = tokens.shape
            tokens = einops.rearrange(tokens, 'b h w -> (b h w)')

            images = [
                transforms(convert_rgb(dataset.access_layer[id_]))
                for id_ in data['id_']
            ]
            patches = einops.rearrange(
                torch.stack(images),
                'b c (ny h) (nx w) -> (b ny nx) c h w',
                ny=ny,
                nx=nx,
            )
            for token, patch in zip(tokens, patches):
                token2patches[token.item()].append(patch)

        todd.logger.info("Saving token2patches")
        torch.save(token2patches, out / 'token2patches.pth')
        todd.logger.info("Saving token2patches done")

    for token, patches in tqdm.tqdm(
        token2patches.items(),
        "Visualizing patches",
    ):
        image = torchvision.utils.make_grid(
            patches[:36],
            6,
            4,
            pad_value=255.,
        )
        image = einops.rearrange(image, 'c h w -> h w c')
        assert cv2.imwrite(
            str(out / f'{token}.png'),
            cv2.cvtColor(image.cpu().numpy(), cv2.COLOR_RGB2BGR),
        )


if __name__ == '__main__':
    main()
