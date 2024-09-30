_base_ = [
    'vanilla.py',
]

trainer = dict(
    dataset=dict(
        name='hq_faces_train',
        num_categories=1,
        access_layer=dict(
            type='ConcatAccessLayer',
            access_layers=dict(
                celeba_hq=dict(
                    type='PILAccessLayer',
                    data_root='data/celeba-hq-256/',
                    task_name='train',
                ),
                ffhq=dict(
                    type='PILAccessLayer',
                    data_root='data/ffhq-256/',
                    task_name='train',
                ),
            ),
        ),
    ),
)
validator = dict(
    dataset=dict(
        name='hq_faces_val',
        num_categories=1,
        access_layer=dict(
            type='ConcatAccessLayer',
            access_layers=dict(
                celeba_hq=dict(
                    type='PILAccessLayer',
                    data_root='data/celeba-hq-256/',
                    task_name='val',
                ),
                ffhq=dict(
                    type='PILAccessLayer',
                    data_root='data/ffhq-256/',
                    task_name='val',
                ),
            ),
        ),
    ),
)
