from typing import Any

_kwargs_: dict[str, Any]
_kwargs_ = dict(_kwargs_)
model = _kwargs_['model']

_base_ = [
    f'{model}.py',
]

_export_: dict[str, Any] = dict()
