"""Interface for pixel decoders.

Example:
    bash tools/torchrun.sh -m vq.train decoder/vqgan/\
vqkd_clip_8192_vqgan_8192_dd2_aglwg075_imagenet_ddp configs/decoder/vqgan.py \
--config-options it_config::configs/vqkd/clip_8192_imagenet_ddp.py decoder::\
vqkd --load-model-from work_dirs/vqkd/clip_8192_imagenet_ddp/checkpoints/\
iter_1/model.pth
"""

from typing import Any

from todd.configs import PyConfig

_kwargs_: dict[str, Any]
_kwargs_ = dict(_kwargs_)
it_config = _kwargs_['it_config']
ir_config = _kwargs_['ir_config']
decoder = _kwargs_.get('decoder')

it = PyConfig.load(it_config, **_kwargs_)
ir = PyConfig.load(ir_config, **_kwargs_)

_base_ = [ir]

it_model = it.validator.model
model = dict(
    encoder=dict(_delete_=True, **it_model.encoder),
    post_encode=dict(_delete_=True, **it_model.post_encode),
    quantizer=dict(_delete_=True, **it_model.quantizer),
    freeze=dict(
        type='NamedModulesFilter',
        names=('_encoder', '_post_encode', '_quantizer'),
    ),
    filter_state_dict=True,
)

if decoder is not None:
    decoder_config = PyConfig.load(
        f'configs/decoder/{decoder}.py',
        **_kwargs_,
    )
    model['decoder'] = dict(_delete_=True, **decoder_config)

params = dict(
    type='NamedParametersFilter',
    modules=dict(type='NamedModulesFilter', names=('_decoder', '_pre_decode')),
)

_export_ = dict(
    trainer=dict(
        model=model,
        optimizers=dict(generator=dict(params=[dict(params=params)])),
    ),
    validator=dict(model=model),
    custom_imports=ir.custom_imports + it.custom_imports,
)
