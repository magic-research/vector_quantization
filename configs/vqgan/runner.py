# pylint: disable=invalid-name

VQITMetricRegistry = 'VQMetricRegistry.VQITMetricRegistry'
VQIRMetricRegistry = 'VQMetricRegistry.VQIRMetricRegistry'
VQIRLossRegistry = 'VQLossRegistry.VQIRLossRegistry'

optimize_callback = dict(
    type='OptimizeCallback',
    # grad_clipper=dict(type='clip_grad_norm_', max_norm=1.0),
)

trainer = dict(
    type='VQGANTrainer',
    callbacks=[
        optimize_callback,
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
    generator_optimize_callback=optimize_callback,
    optimizers=dict(
        generator=dict(
            type='Adam',
            params=[
                dict(
                    params=dict(
                        type='NamedParametersFilter',
                        modules=dict(
                            type='NamedModulesFilter',
                            names=(
                                '_encoder',
                                '_decoder',
                                '_quantizer',
                                '_post_encode',
                                '_pre_decode',
                            ),
                        ),
                    ),
                ),
            ],
            lr=5.4e-5,
            betas=(0.5, 0.9),
        ),
        discriminator=dict(
            type='Adam',
            params=[
                dict(
                    params=dict(
                        type='NamedParametersFilter',
                        modules=dict(
                            type='NamedModulesFilter',
                            name='_discriminator',
                        ),
                    ),
                ),
            ],
            lr=4.32e-4,
            betas=(0.5, 0.9),
        ),
    ),
    iters=26e4,
)
validator = dict(
    type='BaseValidator',
    callbacks=[
        dict(
            type='MetricCallback',
            metrics=dict(
                lpips_loss=dict(
                    type='LossMetric',
                    loss=dict(
                        type='VQLossRegistry.VQIRLossRegistry.LPIPSLoss',
                    ),
                    inputs=dict(
                        pred_image='["pred_image"]',
                        image='["image"]',
                    ),
                ),
                l1_image_loss=dict(
                    type='VQMetricRegistry.ImageLossMetric',
                    loss=dict(type=f'{VQIRLossRegistry}.L1Loss'),
                    pred_image='["pred_image"]',
                    image='["image"]',
                ),
                mse_image_loss=dict(
                    type='VQMetricRegistry.ImageLossMetric',
                    loss=dict(type=f'{VQIRLossRegistry}.MSELoss'),
                    pred_image='["pred_image"]',
                    image='["image"]',
                ),
                psnr=dict(
                    type='VQMetricRegistry.ImageLossMetric',
                    loss=dict(type='VQLossRegistry.VQIRLossRegistry.PSNRLoss'),
                    pred_image='["pred_image"]',
                    image='["image"]',
                ),
                ssim=dict(
                    type='VQMetricRegistry.ImageLossMetric',
                    loss=dict(type='VQLossRegistry.VQIRLossRegistry.SSIMLoss'),
                    pred_image='["pred_image"]',
                    image='["image"]',
                ),
                fid=dict(
                    type='VQMetricRegistry.FIDMetric',
                    pred='["pred_image"]',
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
            eta=dict(type='EMA_ETA', ema=dict(decay=0.9)),
            with_file_handler=True,
        ),
        dict(type='VQCallbackRegistry.BatchedVisualCallback', interval=500),
    ],
)

_export_ = dict(trainer=trainer, validator=validator)
