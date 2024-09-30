from typing import Any

from todd.configs import PyConfig

_kwargs_: dict[str, Any]
_kwargs_ = dict(_kwargs_)

_kwargs_.setdefault('embedding_dim', 768)

_base_ = [
    PyConfig.load('configs/vq/embedding_dim.py', **_kwargs_),
]

teacher = dict(
    type='ViTTeacher',
    model=dict(type='vit_b_16', weights='.ViT_B_16_Weights.IMAGENET1K_V1'),
    downsample_factor=16,
)
runner = dict(model=dict(encoder=dict(teacher=teacher)))

_export_ = dict(trainer=runner, validator=runner)
