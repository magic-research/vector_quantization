from typing import Any

from todd.configs import PyConfig

_kwargs_: dict[str, Any]
_kwargs_ = dict(_kwargs_)

_kwargs_.setdefault('batch_sizes', (512, 512))
_kwargs_.setdefault('num_workers', (10, 10))
_kwargs_.setdefault('image_size', 224)
_kwargs_.setdefault('augmentation', 'default')
_kwargs_.setdefault('batch_size_in_total', True)

_base_ = [
    PyConfig.load('configs/datasets/interface.py', **_kwargs_),
    '../strategies/interface.py',
    'model.py',
    'runner.py',
    'custom_imports.py',
]

_export_: dict[str, Any] = dict()
