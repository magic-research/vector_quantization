# pylint: disable=invalid-name

from typing import Any

_kwargs_: dict[str, Any]
_kwargs_ = dict(_kwargs_)
iters = _kwargs_.get('iters', 250_000)
warmup_iters = _kwargs_.get('warmup_iters', 25_000)

no_weight_decay_regexes = [
    r'^_(en|de)coder\.(cls_token|pos_embed|patch_embed\.proj\.bias|fc_norm)',
    r'^_(en|de)coder\.blocks\.\d+\.norm',
    r'^_(en|de)coder\.blocks\.\d+\.attn\.([qv]_|proj\.)bias$',
    r'^_(en|de)coder\.blocks\.\d+\.mlp\.fc[12]\.bias$',
    r'^_(en|de)coder\.task_layer\.[02]\.bias$',
]
weight_decay_regexes = [
    r'^_(en|de)coder\.patch_embed\.proj\.weight$',
    r'^_(en|de)coder\.blocks\.\d+\.attn\.(qkv|proj)\.weight$',
    r'^_(en|de)coder\.blocks\.\d+\.mlp\.fc[12]\.weight$',
    r'^_(en|de)coder\.task_layer\.[02]\.weight$',
]
params = [
    dict(
        params=dict(type='NamedParametersFilter', regex=regex),
        weight_decay=0.0,
    ) for regex in no_weight_decay_regexes
] + [
    dict(params=dict(type='NamedParametersFilter', regex=regex))
    for regex in weight_decay_regexes
]

VQITMetricRegistry = 'VQMetricRegistry.VQITMetricRegistry'

trainer = dict(
    type='BaseTrainer',
    callbacks=[
        dict(type='OptimizeCallback'),
        dict(
            type='LRScheduleCallback',
            lr_scheduler=dict(
                type='SequentialLR',
                schedulers=[
                    dict(
                        type='LinearLR',
                        start_factor=1e-8,
                        total_iters=warmup_iters - 1,
                    ),
                    dict(
                        type='CosineAnnealingLR',
                        T_max=iters - warmup_iters,
                        eta_min=1e-5,
                    ),
                ],
                milestones=[warmup_iters],
            ),
        ),
        dict(
            type='LogCallback',
            interval=50,
            collect_env=dict(),
            with_file_handler=True,
            eta=dict(type='EMA_ETA', ema=dict(decay=0.9)),
            priority=dict(init=-1),
        ),
        dict(type='GitCallback', diff='HEAD'),
        dict(
            type='TensorBoardCallback',
            interval=50,
            summary_writer=dict(),
            main_tag='train',
        ),
        dict(type='CheckpointCallback', interval=1e4),
    ],
    optimizer=dict(
        type='AdamW',
        lr=2e-4,
        weight_decay=1e-4,
        betas=(0.9, 0.99),
        eps=1e-8,
        params=params,
    ),
    iters=iters,
)
validator = dict(
    type='BaseValidator',
    callbacks=[
        dict(
            type='MetricCallback',
            metrics=dict(
                cosine_embedding_r_loss=dict(
                    type='ReadyMadeMetric',
                    attr='["losses"]["r_loss"]',
                ),
                codebook_usage=dict(
                    type=f'{VQITMetricRegistry}.CodebookUsageMetric',
                    quant='["quantizer"]["quant"]',
                ),
                codebook_ppl=dict(
                    type=f'{VQITMetricRegistry}.CodebookPPLMetric',
                    quant='["quantizer"]["quant"]',
                ),
            ),
        ),
        dict(
            type='LogCallback',
            interval=50,
            collect_env=dict(),
            with_file_handler=True,
            eta=dict(type='EMA_ETA', ema=dict(decay=0.9)),
        ),
    ],
)

_export_ = dict(trainer=trainer, validator=validator)
