# pylint: disable=invalid-name

from typing import Any

from todd.configs import PyConfig

_kwargs_: dict[str, Any]
_kwargs_ = dict(_kwargs_)

_kwargs_.setdefault('distance', 'Cosine')

_base_ = [
    PyConfig.load(
        'configs/vqgan/8192_dd2_aglwg075_imagenet_ddp.py',
        **_kwargs_,
    ),
    'custom_imports.py',
    'quantizer.py',
]

_export_: dict[str, Any] = dict()
