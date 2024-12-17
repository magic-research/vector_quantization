# pylint: disable=import-outside-toplevel

import argparse
import os
import pathlib
import subprocess  # nosec B404
import sys
from typing import Any, Callable
import unittest.mock

import todd
import torch
from todd.patches.py_ import get_

GITHUB = 'https://github.com/'
# GITHUB = 'https://github.moeyy.xyz/https://github.com/'


def download_state_dict(url: str, path: pathlib.Path) -> None:
    if path.exists():
        return
    todd.logger.info("downloading %s to %s", url, path)
    torch.hub.load_state_dict_from_url(
        url,
        str(path.parent),
        'cpu',
        file_name=path.name,
    )


def taming_transformers(args: argparse.Namespace) -> None:
    # Mock pytorch_lightning to load the checkpoint
    sys.modules['pytorch_lightning'] = unittest.mock.Mock()
    sys.modules['pytorch_lightning.callbacks.model_checkpoint'] = \
        unittest.mock.Mock()

    root: pathlib.Path = args.root / 'taming-transformers'
    root.mkdir(parents=True, exist_ok=True)

    # https://colab.research.google.com/drive/1_JqtXRpm25dlGYM7VXvBehyJE5X8xptP?usp=sharing
    download_state_dict(
        'https://heibox.uni-heidelberg.de/d/8088892a516d4e3baf92/files/'
        '?p=%2Fckpts%2Flast.ckpt&dl=1',
        root / 'vqgan_imagenet_f16_1024.pth',
    )
    download_state_dict(
        'https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/'
        '?p=/ckpts/last.ckpt&dl=1',
        root / 'vqgan_imagenet_f16_16384.pth',
    )


def beitv2(args: argparse.Namespace) -> None:
    root: pathlib.Path = args.root / 'beitv2'
    root.mkdir(parents=True, exist_ok=True)

    download_state_dict(
        f'{GITHUB}addf400/files/releases/download/BEiT-v2/'
        'vqkd_encoder_base_decoder_1x768x12_clip-d93179da.pth',
        root / 'vqkd_encoder_base_decoder_1x768x12_clip.pth',
    )
    download_state_dict(
        f'{GITHUB}addf400/files/releases/download/BEiT-v2/'
        'vqkd_encoder_base_decoder_3x768x12_clip-d5036aa7.pth',
        root / 'vqkd_encoder_base_decoder_3x768x12_clip.pth',
    )
    download_state_dict(
        f'{GITHUB}addf400/files/releases/download/BEiT-v2/'
        'vqkd_encoder_base_decoder_1x768x12_dino-663c55d7.pth',
        root / 'vqkd_encoder_base_decoder_1x768x12_dino.pth',
    )


def dino(args: argparse.Namespace) -> None:
    root: pathlib.Path = args.root / 'dino'
    root.mkdir(parents=True, exist_ok=True)

    download_state_dict(
        'https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/'
        'dino_vitbase16_pretrain.pth',
        root / 'vitbase16.pth',
    )


def lpips(args: argparse.Namespace) -> None:
    root: pathlib.Path = args.root / 'lpips'
    root.mkdir(parents=True, exist_ok=True)

    download_state_dict(
        'https://heibox.uni-heidelberg.de/f/607503859c864bc1b30b/?dl=1',
        root / 'vgg.pth',
    )


def pytorch_fid(args: argparse.Namespace) -> None:
    root: pathlib.Path = args.root / 'pytorch-fid'
    root.mkdir(parents=True, exist_ok=True)

    download_state_dict(
        f'{GITHUB}mseitzer/pytorch-fid/releases/download/'
        'fid_weights/pt_inception-2015-12-05-6726825d.pth',
        root / 'pt_inception.pth',
    )


def mae(args: argparse.Namespace) -> None:
    root: pathlib.Path = args.root / 'mae'
    root.mkdir(parents=True, exist_ok=True)

    download_state_dict(
        'https://dl.fbaipublicfiles.com/mae/pretrain/'
        'mae_pretrain_vit_base.pth',
        root / 'mae_pretrain_vit_base.pth',
    )


def clip(args: argparse.Namespace) -> None:
    from clip import load

    root: pathlib.Path = args.root / 'clip'
    root.mkdir(parents=True, exist_ok=True)

    for w in args.weights:
        load(w, download_root=root)


def torchvision(args: argparse.Namespace) -> None:
    from torchvision import models

    root: pathlib.Path = args.root / 'torch'
    if not root.exists():
        root.symlink_to(os.path.join(torch.hub.get_dir(), 'checkpoints'), True)

    for w in args.weights:
        weights: models.WeightsEnum = get_(models, w)
        weights.get_state_dict()


def inception(args: argparse.Namespace) -> None:
    # Inception weights ported to Pytorch from
    # http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
    torch.hub.load_state_dict_from_url(
        f'{GITHUB}toshas/torch-fidelity/releases/download/v0.2.0/'
        'weights-inception-2015-12-05-6726825d.pth',
    )


def huggingface(args: argparse.Namespace) -> None:
    root: pathlib.Path = args.root / 'huggingface'
    root.mkdir(parents=True, exist_ok=True)

    w: str
    for w in args.weights:
        dir_ = root / w
        if not dir_.exists():
            subprocess.run(  # nosec B603 B607
                [
                    'git',
                    'clone',
                    'https://huggingface.co/gpt2-medium',
                    str(dir_),
                ],
                check=True,
                env=dict(os.environ, GIT_LFS_SKIP_SMUDGE='1'),
            )
            subprocess.run(  # nosec B603 B607
                [
                    'git',
                    '-C',
                    str(dir_),
                    'lfs',
                    'pull',
                    '-I',
                    'model.safetensors',
                ],
                check=True,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Prepare checkpoints')
    parser.add_argument(
        '--root',
        type=pathlib.Path,
        default=pathlib.Path('pretrained'),
    )

    subparsers = parser.add_subparsers()

    def add_parser(f: Callable[..., Any]) -> argparse.ArgumentParser:
        subparser = subparsers.add_parser(f.__name__)
        subparser.set_defaults(func=f)
        return subparser

    add_parser(taming_transformers)
    add_parser(beitv2)
    add_parser(dino)
    add_parser(lpips)
    add_parser(pytorch_fid)
    add_parser(mae)

    parser_clip = add_parser(clip)
    parser_clip.add_argument(
        '--weights',
        nargs='+',
        default=[
            'ViT-B/32',
            'ViT-B/16',
            'ViT-L/14',
        ],
    )

    parser_torchvision = add_parser(torchvision)
    parser_torchvision.add_argument(
        '--weights',
        nargs='+',
        default=[
            '.VGG16_Weights.DEFAULT',
            '.Inception_V3_Weights.DEFAULT',
            '.ViT_B_16_Weights.DEFAULT',
            '.ResNet50_Weights.DEFAULT',
            # '.ConvNeXt_Base_Weights.DEFAULT',
        ],
    )

    add_parser(inception)

    parser_huggingface = add_parser(huggingface)
    parser_huggingface.add_argument(
        '--weights',
        nargs='+',
        default=[
            'gpt2-medium',
        ],
    )

    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
