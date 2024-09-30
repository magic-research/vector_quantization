from typing import Any

from todd.configs import PyConfig

_kwargs_: dict[str, Any]
_kwargs_ = dict(_kwargs_)

_kwargs_.setdefault('batch_sizes', (512, 64))
_kwargs_.setdefault('num_workers', (10, 5))
_kwargs_.setdefault('image_size', 256)
_kwargs_.setdefault('augmentation', 'strong')

_base_ = [
    PyConfig.load('configs/datasets/interface.py', **_kwargs_),
    '../strategies/interface.py',
    'model.py',
    'runner.py',
    'custom_imports.py',
]

model = dict(
    type='VQModelRegistry.VQICModelRegistry.BaseModel',
    freeze=dict(type='NamedModulesFilter', name='_it'),
    filter_state_dict=True,
    num_categories=1000,
)
runner = dict(model=model)

_export_ = dict(trainer=runner, validator=runner)
