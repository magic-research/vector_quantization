import argparse
import datetime
import pathlib
from typing import Any, Mapping

import todd
import torch
import torch.utils.data
import torch_fidelity
from todd.configs import PyConfig
from todd.patches.py_ import DictAction
from todd.registries import DatasetRegistry

import vq


class Dataset(torch.utils.data.Dataset[torch.Tensor]):

    def __init__(self, *args, dataset: vq.datasets.Dataset, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert dataset.image_size == 256
        self._dataset = dataset

    @property
    def dataset(self) -> vq.datasets.Dataset:
        return self._dataset

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> torch.Tensor:
        item = self._dataset[index]
        return self._dataset.decode(item['image'])


class Dataset2(Dataset):

    def __init__(
        self,
        *args,
        dataset: str,
        train: bool,
        override: Mapping[str, Any],
        **kwargs,
    ) -> None:
        config: PyConfig = PyConfig.load(
            'configs/datasets/interface.py',
            dataset=dataset,
            augmentation='none',
        )
        config = config.trainer if train else config.validator
        config = config.dataset
        config.override(override)
        super().__init__(
            *args,
            dataset=DatasetRegistry.build(config),
            **kwargs,
        )


class Dataset1(Dataset):

    def __init__(
        self,
        *args,
        dataset: vq.datasets.Dataset,
        data_root: str,
        override: Mapping[str, Any],
        **kwargs,
    ) -> None:
        transforms = dataset.transforms
        assert transforms is not None

        config = todd.Config(
            type='VQDatasetRegistry.Dataset',
            name=self.__class__.__name__.lower(),
            num_categories=1,
            access_layer=dict(
                type='PILAccessLayer',
                data_root=data_root,
                subfolder_action='none',
                suffix='png',
            ),
            image_size=dataset.image_size,
            transforms=transforms.transforms,
        )
        config.override(override)
        dataset = DatasetRegistry.build(config)
        super().__init__(*args, dataset=dataset, **kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    parser.add_argument('data_root')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--override1', action=DictAction, default=dict())
    parser.add_argument('--override2', action=DictAction, default=dict())
    parser.add_argument('--metrics', nargs='+', default=['fid', 'isc'])
    parser.add_argument('--work-dir', type=pathlib.Path)
    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()
    dataset2 = Dataset2(  # reference
        dataset=args.dataset,
        train=args.train,
        override=args.override1,
    )
    dataset1 = Dataset1(  # predicted
        dataset=dataset2.dataset,
        data_root=args.data_root,
        override=args.override2,
    )

    # IS metrics are evaluated for dataset1, so dataset1 must not be reference
    metrics = torch_fidelity.calculate_metrics(
        input1=dataset1,
        input2=dataset2,
        input2_cache_name=dataset2.dataset.name,
        **{k: True
           for k in args.metrics},
    )

    if args.work_dir is not None:
        work_dir: pathlib.Path = args.work_dir
        output_file = work_dir / 'torch_fidelity.txt'
        with output_file.open('a') as f:
            f.write(f"{datetime.datetime.now()}\n")
            f.write(f"{args=}\n")
            f.write(f"{metrics=}\n")


if __name__ == '__main__':
    main()
