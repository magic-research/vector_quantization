_base_ = [
    'vqgan_imagenet_ddp.py',
]

discriminator = dict(
    _delete_=True,
    type='StyleGAN2Discriminator',
    image_size=256,
)
runner = dict(model=dict(discriminator=discriminator))

_export_ = dict(trainer=runner, validator=runner)
