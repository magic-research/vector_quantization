from todd.configs import PyConfig

_export_ = PyConfig.load(
    'configs/cluster/interface.py',
    encoder='clip',
    num_embeddings=8192,
    dataset='imagenet',
    strategy='ddp',
)
