decoder = dict(out_chans=768)
distiller = dict(
    teacher=dict(
        type='MAETeacher',
        model=dict(
            type='mae_vit_base_patch16',
            init_weights=dict(
                pretrained='pretrained/mae/mae_pretrain_vit_base.pth',
            ),
        ),
        downsample_factor=16,
    ),
)
