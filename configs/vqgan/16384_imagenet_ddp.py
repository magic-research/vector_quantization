# pylint: disable=invalid-name

from todd.configs import PyConfig

_export_ = PyConfig.load(
    'configs/vqgan/interface.py',
    num_embeddings=16384,
    dataset='imagenet',
    strategy='ddp',
    find_unused_parameters=True,
)
