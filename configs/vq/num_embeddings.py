from typing import Any

_kwargs_: dict[str, Any]
_kwargs_ = dict(_kwargs_)
num_embeddings = _kwargs_['num_embeddings']

runner = dict(
    model=dict(quantizer=dict(embedding=dict(num_embeddings=num_embeddings))),
)

_export_ = dict(trainer=runner, validator=runner)
