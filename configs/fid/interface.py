from typing import Any

from todd.configs import PyConfig

_kwargs_: dict[str, Any]
_kwargs_ = dict(_kwargs_)

_kwargs_.setdefault('batch_sizes', (128, 128))
_kwargs_.setdefault('num_workers', (10, 10))
_kwargs_.setdefault('image_size', 256)
_kwargs_.setdefault('augmentation', 'none')

_base_ = [
    PyConfig.load('configs/datasets/interface.py', **_kwargs_),
    '../strategies/interface.py',
]

runner = dict(
    type='VQRunnerRegistry.BaseValidator',
    model=dict(type='FIDModel'),
    callbacks=[
        dict(
            type='LogCallback',
            interval=20,
            collect_env=dict(),
            eta=dict(type='EMA_ETA', ema=dict(decay=0.9)),
            with_file_handler=True,
        ),
        dict(type='FIDCallback'),
    ],
)

_export_ = dict(trainer=runner, validator=runner)
