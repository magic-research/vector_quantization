from typing import Any

from todd.configs import PyConfig

_kwargs_: dict[str, Any]
_kwargs_ = dict(_kwargs_)

_kwargs_.setdefault('embedding_dim', 32)
_kwargs_.setdefault('distance', 'Cosine')

_base_ = [
    PyConfig.load('configs/vq/interface.py', **_kwargs_),
    'teachers/interface.py',
]

model = dict(
    type='VQModelRegistry.VQKD',
    encoder=dict(type='VQKDEncoder'),
    post_encode=dict(type='BaseConnector'),
    quantizer=dict(
        type='VQKDQuantizer',
        callbacks=[dict(type='VQKDCallback', ema=dict())],
        losses=dict(
            commitment_loss=dict(type='CommitmentLoss', mse=dict(norm=True)),
        ),
    ),
    pre_decode=dict(type='BaseConnector'),
    decoder=dict(type='VQKDDecoder', depth=1),
    distiller=dict(
        type='VQKDDistiller',
        teacher=dict(output_wh=(14, 14)),
        teacher_hook_pipeline=dict(
            target_features=dict(
                type='SingleOperator',
                args=tuple(),
                atom=dict(
                    type=(
                        'TaskRegistry.KDRegistry.KDDistillerRegistry.'
                        'KDHookRegistry.Hook'
                    ),
                    path='',
                ),
            ),
        ),
        student_hook_pipeline=dict(),
        adapt_pipeline=dict(
            pred_features=dict(
                type='SingleOperator',
                args=('pred_features', ),
                atom=dict(
                    type=(
                        'TaskRegistry.KDRegistry.KDDistillerRegistry.'
                        'KDAdaptRegistry.Model'
                    ),
                    model=dict(
                        type='einops_layers_torch_Rearrange',
                        pattern='b c h w -> b (h w) c',
                    ),
                ),
            ),
        ),
        loss_pipeline=dict(
            r_loss=dict(
                type='SingleOperator',
                args=('pred_features', 'target_features'),
                atom=dict(
                    type=(
                        'ModelRegistry.LossRegistry.VQLossRegistry.'
                        'CosineEmbeddingLoss'
                    ),
                    cosine_embedding=dict(),
                ),
            ),
        ),
    ),
    no_grad=dict(
        type='NamedParametersFilter',
        modules=dict(
            type='NamedModulesFilter',
            name='_quantizer',
        ),
    ),
)
runner = dict(model=model)

_export_ = dict(trainer=runner, validator=runner)
