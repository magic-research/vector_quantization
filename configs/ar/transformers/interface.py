from typing import Any

from todd.configs import PyConfig

_kwargs_: dict[str, Any]
_kwargs_ = dict(_kwargs_)
transformer = _kwargs_['transformer']

transformer = PyConfig.load(
    f'configs/ar/transformers/{transformer}.py',
    **_kwargs_,
)
runner = dict(model=dict(transformer=transformer))

_export_ = dict(trainer=runner, validator=runner)
