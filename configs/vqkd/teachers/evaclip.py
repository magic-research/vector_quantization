decoder = dict(out_chans=768)
distiller = dict(
    teacher=dict(
        type='EVACLIPTeacher',
        model=dict(model_name='EVA02-CLIP-B-16', pretrained='eva02_clip'),
        downsample_factor=16,
        image_wh=(224, 224),
    ),
)
