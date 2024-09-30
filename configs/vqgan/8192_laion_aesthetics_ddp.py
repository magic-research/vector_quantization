# pylint: disable=invalid-name

from todd.configs import PyConfig

_export_ = PyConfig.load(
    'configs/vqgan/interface.py',
    num_embeddings=8192,
    dataset='laion_aesthetics',
    strategy='ddp',
    find_unused_parameters=True,
)
