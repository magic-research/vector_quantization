decoder = dict(out_chans=768)
distiller = dict(
    teacher=dict(
        type='ViTTeacher',
        model=dict(type='vit_b_16', weights='.ViT_B_16_Weights.IMAGENET1K_V1'),
        downsample_factor=16,
    ),
)
