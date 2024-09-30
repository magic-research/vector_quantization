from typing import Any

from todd.configs import PyConfig

_kwargs_: dict[str, Any]
_kwargs_ = dict(_kwargs_)

_kwargs_.setdefault('embedding_dim', 768)

_base_ = [
    PyConfig.load('configs/vq/embedding_dim.py', **_kwargs_),
]

teacher = dict(
    type='MAETeacher',
    model=dict(
        type='mae_vit_base_patch16',
        init_weights=dict(
            pretrained='pretrained/mae/mae_pretrain_vit_base.pth',
        ),
    ),
    downsample_factor=16,
)
runner = dict(model=dict(encoder=dict(teacher=teacher)))

_export_ = dict(trainer=runner, validator=runner)
