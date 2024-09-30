dataset_type = 'VQDatasetRegistry.COCODataset'  # pylint: disable=invalid-name

_export_ = dict(
    trainer=dict(
        dataset=dict(
            type=dataset_type,
            name='coco_2014_train',
            num_categories=80,
            split='train',
            year=2014,
        ),
    ),
    validator=dict(
        dataset=dict(
            type=dataset_type,
            name='coco_2014_val',
            num_categories=80,
            split='val',
            year=2014,
        ),
    ),
)
