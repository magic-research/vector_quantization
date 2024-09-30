from typing import Any

from todd.configs import PyConfig

_kwargs_: dict[str, Any]
_kwargs_ = dict(_kwargs_)

_kwargs_.setdefault('batch_sizes', (128, 128))
_kwargs_.setdefault('num_workers', (10, 10))
_kwargs_.setdefault('num_embeddings', 16384)
_kwargs_.setdefault('embedding_dim', 8)

_base_ = [
    PyConfig.load('configs/vqgan/interface.py', **_kwargs_),
]

model = dict(
    quantizer=dict(callbacks=[
        dict(type='NormalizeCallback'),
    ]),
    reconstruct_losses=dict(
        l1_r_loss=None,
        l2_r_loss=dict(type='MSELoss'),
    ),
    generator_loss=dict(weight=0.5),
    discriminator_loss=dict(weight=0.5),
    adaptive_generator_loss_weight_gain=None,
)
trainer = dict(
    model=model,
    discriminator_start=20_000,
    optimizers=dict(
        generator=dict(lr=1e-4, betas=(0.9, 0.95)),
        discriminator=dict(lr=1e-4, betas=(0.9, 0.95)),
    ),
    iters=400_000,
)
validator = dict(model=model)

_export_ = dict(trainer=trainer, validator=validator)
