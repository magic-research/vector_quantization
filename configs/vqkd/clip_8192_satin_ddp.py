from todd.configs import PyConfig

_export_ = PyConfig.load(
    'configs/vqkd/interface.py',
    teacher='clip',
    num_embeddings=8192,
    dataset='satin',
    strategy='ddp',
)
