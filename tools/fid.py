import argparse
import datetime
import pathlib

import todd.tasks.image_generation as ig
from todd.configs import PyConfig
from todd.patches.py import DictAction
from todd.patches.torch import get_rank, load
from todd.registries import RunnerRegistry
from torch import nn

from vq.runners import BaseValidator
from vq.utils import Store, log


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('reference')
    parser.add_argument('data_root')
    parser.add_argument('--strategy', default='cuda')
    parser.add_argument('--override', action=DictAction, default=dict())
    parser.add_argument('--autocast', action='store_true')
    parser.add_argument('--work-dir', type=pathlib.Path)
    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()
    reference: ig.Statistics = load(
        f'pretrained/fid/{args.reference}.pth',
        directory=Store.PRETRAINED,
    )

    config: PyConfig = PyConfig.load(
        'configs/fid/interface.py',
        dataset='vanilla',
        strategy=args.strategy,
    )
    config = config.validator

    if args.work_dir is None:
        fid_path = '/dev/null'
    else:
        work_dir: pathlib.Path = args.work_dir
        fid_path = args.work_dir / 'fid.pth'

    config.update(
        dataset=dict(
            name=__file__.replace('/', '_'),
            num_categories=1,
            fid_path=fid_path,
            access_layer=dict(
                data_root=args.data_root,
                subfolder_action='none',
                suffix='png',
            ),
        ),
    )
    config.override(args.override)

    name: str = args.data_root
    name = name.replace('/', '_')

    runner: BaseValidator[nn.Module] = RunnerRegistry.build(
        config,
        name=f'fid/{name}',
        autocast=args.autocast,
    )
    log(runner, args, config)
    memo = runner.run()

    fid = ig.fid(reference, memo['statistics'])

    if get_rank() == 0:
        runner.logger.info(f'FID: {fid}')
        if args.work_dir is not None:
            output_file = work_dir / 'fid.txt'
            with open(output_file, 'a') as f:
                f.write(f"{datetime.datetime.now()}\n")
                f.write(f"{args=}\n")
                f.write(f"{fid=}\n")


if __name__ == '__main__':
    main()
