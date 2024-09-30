from todd.configs import PyConfig

_export_ = PyConfig.load(
    'configs/llamagen/vqgan.py',
    dataset='imagenet',
    strategy='ddp',
    find_unused_parameters=True,
    image_size=384,
)
