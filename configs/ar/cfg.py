from typing import Any

_kwargs_: dict[str, Any]
_kwargs_ = dict(_kwargs_)
dropout = _kwargs_.get('dropout', 0.1)
alpha = _kwargs_.get('alpha', 1.75)

_export_ = dict(
    trainer=dict(model=dict(cfg=dropout)),
    validator=dict(
        model=dict(cfg=dropout, transformer=dict(sampler=dict(cfg=alpha))),
    ),
)
