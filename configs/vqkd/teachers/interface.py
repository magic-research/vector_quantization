from typing import Any

from todd.configs import PyConfig

_kwargs_: dict[str, Any]
_kwargs_ = dict(_kwargs_)
teacher = _kwargs_['teacher']

model = PyConfig.load(f'configs/vqkd/teachers/{teacher}.py')
runner = dict(model=model)

_export_ = dict(trainer=runner, validator=runner)
