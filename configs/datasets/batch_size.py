from typing import Any

_kwargs_: dict[str, Any]
_kwargs_ = dict(_kwargs_)
trainer_batch_size, validator_batch_size = _kwargs_['batch_sizes']
trainer_num_workers, validator_num_workers = _kwargs_['num_workers']

_export_ = dict(
    trainer=dict(
        dataloader=dict(
            batch_size=trainer_batch_size,
            num_workers=trainer_num_workers,
        ),
    ),
    validator=dict(
        dataloader=dict(
            batch_size=validator_batch_size,
            num_workers=validator_num_workers,
        ),
    ),
)
