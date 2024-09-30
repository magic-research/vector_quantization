from typing import Any

from todd.configs import PyConfig

_kwargs_: dict[str, Any]
_kwargs_ = dict(_kwargs_)

_kwargs_.setdefault('embedding_dim', 256)
_kwargs_.setdefault('distance', 'L2')

_base_ = [
    PyConfig.load('configs/vq/interface.py', **_kwargs_),
]

model = dict(
    type='VQModelRegistry.VQGAN',
    encoder=dict(type='VQGANEncoder'),
    post_encode=dict(type='ConvConnector'),
    quantizer=dict(
        type='VQGANQuantizer',
        losses=dict(vqgan_loss=dict(type='VQGANLoss')),
        init_weights=dict(type='vqgan'),
    ),
    pre_decode=dict(type='ConvConnector'),
    decoder=dict(type='VQGANDecoder'),
    discriminator=dict(type='PatchGANDiscriminator'),
    reconstruct_losses=dict(
        l1_r_loss=dict(type='L1Loss'),
        lpips_r_loss=dict(type='LPIPSLoss'),
    ),
    generator_loss=dict(type='VQGANGeneratorLoss'),
    discriminator_loss=dict(type='VQGANDiscriminatorLoss'),
)
runner = dict(model=model)

_export_ = dict(trainer=runner, validator=runner)
