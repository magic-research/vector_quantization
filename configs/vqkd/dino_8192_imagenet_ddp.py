from todd.configs import PyConfig

_export_ = PyConfig.load(
    'configs/vqkd/interface.py',
    teacher='dino',
    num_embeddings=8192,
    dataset='imagenet',
    strategy='ddp',
)
