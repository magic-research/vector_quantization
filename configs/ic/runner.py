from typing import Any

_kwargs_: dict[str, Any]
_kwargs_ = dict(_kwargs_)
iters = _kwargs_.get('iters', 30_000)

trainer = dict(
    type='BaseTrainer',
    callbacks=[
        dict(type='OptimizeCallback'),
        dict(
            type='LRScheduleCallback',
            lr_scheduler=dict(type='CosineAnnealingLR', T_max=iters),
            interval=1,
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
        dict(type='CheckpointCallback', interval=1e3),
    ],
    optimizer=dict(
        type='LARSOptimizer',
        params=[
            dict(
                params=dict(
                    type='NamedParametersFilter',
                    modules=dict(
                        type='NamedModulesFilter',
                        name='_head',
                    ),
                ),
            ),
        ],
        lr=1.6,
    ),
    iters=iters,
)
validator = dict(
    type='BaseValidator',
    callbacks=[
        dict(
            type='MetricCallback',
            metrics=dict(
                accuracy=dict(
                    type='AccuracyMetric',
                    topk=5,
                    logits='["logits"]',
                    target='["category"]',
                ),
                loss=dict(
                    type='ReadyMadeMetric',
                    attr='["loss"]',
                ),
            ),
        ),
        dict(
            type='LogCallback',
            interval=1,
            collect_env=dict(),
            eta=dict(type='EMA_ETA', ema=dict(decay=0.9)),
            with_file_handler=True,
        ),
    ],
)

_export_ = dict(trainer=trainer, validator=validator)
