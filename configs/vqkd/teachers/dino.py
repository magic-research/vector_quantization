decoder = dict(out_chans=768)
distiller = dict(teacher=dict(
    type='DINOTeacher',
    downsample_factor=16,
))
