# pylint: disable=invalid-name

VQITMetricRegistry = 'VQMetricRegistry.VQITMetricRegistry'
trainer = dict(
    type='BaseTrainer',
    callbacks=[
        dict(type='OptimizeCallback'),
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
        type='Adam',
        lr=5.4e-5,
        betas=(0.5, 0.9),
        params=[
            dict(
                params=dict(
                    type='NamedParametersFilter',
                    modules=dict(type='NamedModulesFilter', name='_quantizer'),
                ),
            ),
        ],
    ),
    iters=1e5,
)
validator = dict(
    type='BaseValidator',
    callbacks=[
        dict(
            type='MetricCallback',
            metrics=dict(
                loss=dict(
                    type='ReadyMadeMetric',
                    attr='["loss"]',
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
