from todd.configs import PyConfig

_export_ = PyConfig.load(
    'configs/vqkd/interface.py',
    teacher='vit',
    num_embeddings=16384,
    dataset='imagenet',
    strategy='ddp',
)
