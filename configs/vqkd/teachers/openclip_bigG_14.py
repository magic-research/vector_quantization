# pylint: disable=invalid-name

_base_ = [
    'openclip.py',
]

decoder = dict(out_chans=1280)
distiller = dict(
    teacher=dict(
        model=dict(model_name='ViT-bigG-14', pretrained='laion2b_s39b_b160k'),
        downsample_factor=14,
        image_wh=(224, 224),
    ),
)
