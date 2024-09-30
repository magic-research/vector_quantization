# pylint: disable=invalid-name

from todd.configs import PyConfig

_export_ = PyConfig.load(
    'configs/fsq/interface.py',
    codebook_size=64000,
    dataset='imagenet',
    strategy='ddp',
    find_unused_parameters=True,
)
