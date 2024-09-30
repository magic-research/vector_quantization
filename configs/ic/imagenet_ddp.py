from typing import Any

from todd.configs import PyConfig

_kwargs_: dict[str, Any]
_kwargs_ = dict(_kwargs_)

_export_ = PyConfig.load(
    'configs/ic/interface.py',
    dataset='imagenet',
    strategy='ddp',
    **_kwargs_,
)
