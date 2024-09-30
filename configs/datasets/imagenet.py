dataset_type = 'VQDatasetRegistry.ImageNetDataset'  # noqa: E501 pylint: disable=invalid-name

_export_ = dict(
    trainer=dict(
        dataset=dict(
            type=dataset_type,
            name='imagenet_train',
            num_categories=1000,
            split='train',
        ),
    ),
    validator=dict(
        dataset=dict(
            type=dataset_type,
            name='imagenet_val',
            num_categories=1000,
            split='val',
        ),
    ),
)
