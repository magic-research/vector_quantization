# pylint: disable=invalid-name

_base_ = [
    'openclip.py',
]

decoder = dict(out_chans=768)
distiller = dict(
    teacher=dict(
        model=dict(model_name='ViT-L-14', pretrained='laion2B_s32B_b82k'),
        downsample_factor=14,
        image_wh=(224, 224),
    ),
)
