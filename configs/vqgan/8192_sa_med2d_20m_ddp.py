# pylint: disable=invalid-name

from todd.configs import PyConfig

_export_ = PyConfig.load(
    'configs/vqgan/interface.py',
    num_embeddings=8192,
    dataset='sa_med2d_20m',
    strategy='ddp',
    find_unused_parameters=True,
)
