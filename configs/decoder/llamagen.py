from typing import Any

from todd.configs import PyConfig

_kwargs_: dict[str, Any]
_kwargs_ = dict(_kwargs_)

_kwargs_.setdefault(
    'ir_config',
    'configs/llamagen/vqgan_imagenet_ddp.py',
)

_export_ = PyConfig.load(
    'configs/decoder/interface.py',
    num_embeddings=8192,
    dataset='imagenet',
    strategy='ddp',
    find_unused_parameters=True,
    **_kwargs_,
)
