# pylint: disable=invalid-name

_base_ = [
    'openclip.py',
]

decoder = dict(out_chans=1024)
distiller = dict(
    teacher=dict(
        model=dict(model_name='ViT-H-14', pretrained='laion2B_s32B_b79k'),
        downsample_factor=14,
        image_wh=(224, 224),
    ),
)
