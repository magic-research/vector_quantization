from typing import Any

from todd.configs import PyConfig

_kwargs_: dict[str, Any]
_kwargs_ = dict(_kwargs_)
codebook_size = _kwargs_['codebook_size']

_base_ = [
    '../sq/interface.py',
    PyConfig.load('configs/vqgan/model.py', num_embeddings=None, **_kwargs_),
]

NUM_SCALARS_PER_CHANNEL = {
    8000: [8, 8, 5, 5, 5],
    64000: [8, 8, 8, 5, 5, 5],
}
model = dict(
    post_encode=[
        dict(type='ConvConnector', out_channels=256),
        dict(type='ConvConnector'),
    ],
    quantizer=dict(
        _delete_=True,
        type='FiniteScalarQuantizer',
        num_scalars_per_channel=NUM_SCALARS_PER_CHANNEL[codebook_size],
    ),
    pre_decode=[
        dict(type='ConvConnector', out_channels=256),
        dict(type='ConvConnector'),
    ],
)
runner = dict(model=model)

_export_ = dict(trainer=runner, validator=runner)
