from typing import Any

from todd.configs import PyConfig

_kwargs_: dict[str, Any]
_kwargs_ = dict(_kwargs_)
encoder = _kwargs_['encoder']

_export_ = PyConfig.load(f'configs/cluster/encoders/{encoder}.py')
