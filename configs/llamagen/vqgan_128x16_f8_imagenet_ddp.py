from typing import Any

from todd.configs import PyConfig

_base_ = [
    PyConfig.load(
        'configs/llamagen/vqgan.py',
        dataset='imagenet',
        strategy='ddp',
        find_unused_parameters=True,
        num_embeddings=128,
        embedding_dim=16,
    ),
    '../vqgan/f8.py',
]

_export_: dict[str, Any] = dict()
