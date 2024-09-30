# pylint: disable=invalid-name

from typing import Any

_kwargs_: dict[str, Any]
_kwargs_ = dict(_kwargs_)
iters = _kwargs_.get('iters', 260_000)

VQITMetricRegistry = 'VQMetricRegistry.VQITMetricRegistry'
VQIRMetricRegistry = 'VQMetricRegistry.VQIRMetricRegistry'
VQSMMetricRegistry = 'VQMetricRegistry.VQSMMetricRegistry'
VQIRLossRegistry = 'VQLossRegistry.VQIRLossRegistry'

trainer = dict(
    type='VQSMRunnerRegistry.BaseTrainer',
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
        dict(type='VQCallbackRegistry.BatchedVisualCallback', interval=500),
        dict(type='CheckpointCallback', interval=1e4),
    ],
    optimizer=dict(
        type='AdamW',
        params=[
            dict(
                params=dict(
                    type='NamedParametersFilter',
                    modules=dict(
                        type='NamedModulesFilter',
                        name='_transformer',
                    ),
                ),
            ),
        ],
        lr=1e-4,
        betas=(0.9, 0.98),
        weight_decay=0.2,
        eps=1e-6,
    ),
    iters=iters,
)
validator = dict(
    type='VQSMRunnerRegistry.BaseValidator',
    callbacks=[
        dict(
            type='MetricCallback',
            metrics=dict(
                reconstruct_lpips_loss=dict(
                    type='LossMetric',
                    loss=dict(
                        type='VQLossRegistry.VQIRLossRegistry.LPIPSLoss',
                    ),
                    inputs=dict(
                        pred_image='["reconstructed_image"]',
                        image='["image"]',
                    ),
                ),
                reconstruct_l1_image_loss=dict(
                    type='VQMetricRegistry.ImageLossMetric',
                    loss=dict(type=f'{VQIRLossRegistry}.L1Loss'),
                    pred_image='["reconstructed_image"]',
                    image='["image"]',
                ),
                reconstruct_mse_image_loss=dict(
                    type='VQMetricRegistry.ImageLossMetric',
                    loss=dict(type=f'{VQIRLossRegistry}.MSELoss'),
                    pred_image='["reconstructed_image"]',
                    image='["image"]',
                ),
                reconstruct_psnr=dict(
                    type='VQMetricRegistry.ImageLossMetric',
                    loss=dict(type='VQLossRegistry.VQIRLossRegistry.PSNRLoss'),
                    pred_image='["reconstructed_image"]',
                    image='["image"]',
                ),
                reconstruct_ssim=dict(
                    type='VQMetricRegistry.ImageLossMetric',
                    loss=dict(type='VQLossRegistry.VQIRLossRegistry.SSIMLoss'),
                    pred_image='["reconstructed_image"]',
                    image='["image"]',
                ),
                loss=dict(
                    type='ReadyMadeMetric',
                    attr='["loss"]',
                ),
                accuracy=dict(
                    type=f'{VQSMMetricRegistry}.AccuracyMetric',
                    pred='["sample"]["image_tokens"]',
                    target='["image_tokens"]',
                ),
                fid=dict(
                    type='VQMetricRegistry.FIDMetric',
                    pred='["generated_image"]',
                ),
                codebook_usage=dict(
                    type=f'{VQITMetricRegistry}.CodebookUsageMetric',
                    quant='["ir"]["encode_to_quant"]["quantizer"]["quant"]',
                ),
                codebook_ppl=dict(
                    type=f'{VQITMetricRegistry}.CodebookPPLMetric',
                    quant='["ir"]["encode_to_quant"]["quantizer"]["quant"]',
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
        dict(type='VQCallbackRegistry.BatchedVisualCallback', interval=10),
    ],
)

_export_ = dict(trainer=trainer, validator=validator)
