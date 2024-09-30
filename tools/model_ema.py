import argparse
import pathlib

import todd
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Extract EMA')
    parser.add_argument('path', type=pathlib.Path)
    parser.add_argument('attr', default='["callbacks"][2]["shadow"]')
    parser.add_argument('--out', default='model_ema.pth')
    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()
    path: pathlib.Path = args.path
    state_dict = torch.load(path, 'cpu')
    model_ema = todd.patches.py.get_(state_dict, args.attr)
    torch.save(model_ema, path.parent / args.out)


if __name__ == '__main__':
    main()
