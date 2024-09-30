from typing import Any

_kwargs_: dict[str, Any]
_kwargs_ = dict(_kwargs_)
embedding_dim = _kwargs_['embedding_dim']

runner = dict(
    model=dict(quantizer=dict(embedding=dict(embedding_dim=embedding_dim))),
)

_export_ = dict(trainer=runner, validator=runner)
