model = dict(
    encoder=dict(width_mults=(1, 1, 2, 4)),
    decoder=dict(width_mults=(4, 2, 1, 1)),
)
runner = dict(model=model)

_export_ = dict(trainer=runner, validator=runner)
