from typing import Any

from todd.configs import PyConfig

_kwargs_: dict[str, Any]
_kwargs_ = dict(_kwargs_)

_kwargs_.setdefault('embedding_dim', 32)
_kwargs_.setdefault('distance', 'Cosine')

_base_ = [
    PyConfig.load('configs/vq/interface.py', **_kwargs_),
    'encoders/interface.py',
]

model = dict(
    type='VQModelRegistry.Cluster',
    encoder=dict(type='ClusterEncoder'),
    post_encode=dict(type='BaseConnector'),
    quantizer=dict(
        type='VQGANQuantizer',
        losses=dict(vqgan_loss=dict(type='CodebookLoss')),
        init_weights=dict(type='vqgan'),
        callbacks=[
            dict(
                type='CVQVAECallback',
                ema=dict(),
                anchor=dict(type='NearestAnchor', sync=True),
            ),
        ],
    ),
    freeze=dict(type='NamedModulesFilter', name='_encoder'),
    filter_state_dict=True,
)
runner = dict(model=model)

_export_ = dict(trainer=runner, validator=runner)
