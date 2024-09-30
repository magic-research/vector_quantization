from typing import Any

from todd.configs import PyConfig

_kwargs_: dict[str, Any]
_kwargs_ = dict(_kwargs_)

_kwargs_.setdefault('batch_sizes', (96, 96))
_kwargs_.setdefault('num_workers', (5, 5))
_kwargs_.setdefault('image_size', 256)
_kwargs_.setdefault('augmentation', 'weak')
_kwargs_.setdefault('batch_size_in_total', True)

_base_ = [
    PyConfig.load('configs/datasets/interface.py', **_kwargs_),
    '../strategies/interface.py',
    'model.py',
    'runner.py',
    'custom_imports.py',
]

_export_: dict[str, Any] = dict()
