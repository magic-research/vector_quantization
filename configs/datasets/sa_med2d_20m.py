dataset_type = 'VQDatasetRegistry.SAMed2DDataset'  # noqa: E501 pylint: disable=invalid-name

_export_ = dict(
    trainer=dict(
        dataset=dict(
            type=dataset_type,
            name='sa_med2d_train',
            num_categories=1,
            split='v1',
            train=True,
        ),
    ),
    validator=dict(
        dataset=dict(
            type=dataset_type,
            name='sa_med2d_val',
            num_categories=1,
            split='v1',
            train=False,
        ),
    ),
)
