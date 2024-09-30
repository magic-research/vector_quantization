from typing import Any

_kwargs_: dict[str, Any]
_kwargs_ = dict(_kwargs_)
dataset = _kwargs_['dataset']
batch_size_in_total = _kwargs_.get('batch_size_in_total', False)

_base_ = [
    f'{dataset}.py',
    'batch_size.py',
    'transforms/interface.py',
]

if batch_size_in_total:
    _base_.append('batch_size_in_total.py')

_export_: dict[str, Any] = dict()
