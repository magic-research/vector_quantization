# pylint: disable=invalid-name

_base_ = [
    '../llamagen/vqgan_imagenet_ddp.py',
]

coder = dict(attention_layer=None, refine_layer=None)
runner = dict(model=dict(encoder=coder, decoder=coder))

_export_ = dict(trainer=runner, validator=runner)
