# pylint: disable=invalid-name

from todd.configs import PyConfig

_export_ = PyConfig.load(
    'configs/vqkd/interface.py',
    teacher='openclip_bigG_14',
    num_embeddings=8192,
    dataset='imagenet',
    strategy='ddp',
)
