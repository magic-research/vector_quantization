from typing import Any

_kwargs_: dict[str, Any]
_kwargs_ = dict(_kwargs_)
strategy = _kwargs_['distance']

type_ = f'{strategy}Distance'
runner = dict(model=dict(quantizer=dict(distance=dict(type=type_))))

_export_ = dict(trainer=runner, validator=runner)
