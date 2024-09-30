decoder = dict(out_chans=1024)
distiller = dict(
    teacher=dict(
        type='ConvNeXtTeacher',
        model=dict(
            type='convnext_base',
            weights='.ConvNeXt_Base_Weights.IMAGENET1K_V1',
        ),
        downsample_factor=(32, 32),
        image_wh=(224, 224),  # TODO: remove this after rebuttal
    ),
)
