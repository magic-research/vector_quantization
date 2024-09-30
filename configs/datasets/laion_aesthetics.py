dataset_type = 'VQDatasetRegistry.LAIONAestheticsDataset'  # noqa: E501 pylint: disable=invalid-name

_export_ = dict(
    trainer=dict(
        dataset=dict(
            type=dataset_type,
            name='laion_aesthetics_train',
            num_categories=1,
            split='v2_6.5plus',
            train=True,
        ),
    ),
    validator=dict(
        dataset=dict(
            type=dataset_type,
            name='laion_aesthetics_val',
            num_categories=1,
            split='v2_6.5plus',
            train=False,
        ),
    ),
)
