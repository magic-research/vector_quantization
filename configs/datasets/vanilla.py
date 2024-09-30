# pylint: disable=invalid-name

dataset_type = 'VQDatasetRegistry.Dataset'
access_layer_type = 'PILAccessLayer'

_export_ = dict(
    trainer=dict(
        dataset=dict(
            type=dataset_type,
            name='vanilla_train',
            access_layer=dict(type=access_layer_type),
        ),
    ),
    validator=dict(
        dataset=dict(
            type=dataset_type,
            name='vanilla_val',
            access_layer=dict(type=access_layer_type),
        ),
    ),
)
