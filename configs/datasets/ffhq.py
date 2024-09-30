_base_ = [
    'vanilla.py',
]

data_root = 'data/ffhq-256/'  # pylint: disable=invalid-name

_export_ = dict(
    trainer=dict(
        dataset=dict(
            name='ffhq_train',
            num_categories=1,
            access_layer=dict(data_root=data_root, task_name='train'),
        ),
    ),
    validator=dict(
        dataset=dict(
            name='ffhq_val',
            num_categories=1,
            access_layer=dict(data_root=data_root, task_name='val'),
        ),
    ),
)
