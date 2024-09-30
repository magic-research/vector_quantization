from typing import Any

from todd.configs import PyConfig

_kwargs_: dict[str, Any]
_kwargs_ = dict(_kwargs_)

_kwargs_.setdefault('batch_sizes', (256, 256))
_kwargs_.setdefault('num_workers', (10, 10))
_kwargs_.setdefault('transformer', 'llama')

_base_ = [
    PyConfig.load('configs/ar/interface.py', **_kwargs_),
    '../ar/cfg.py',
]

model = dict(transformer=dict(sampler=dict(type='BaseSampler')))
trainer = dict(
    model=model,
    optimizers=dict(betas=(0.9, 0.95), weight_decay=0.05),
    iters=250_000,
)
validator = dict(model=model)

_export_ = dict(trainer=trainer, validator=validator)
