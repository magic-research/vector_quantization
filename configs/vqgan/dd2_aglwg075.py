model = dict(
    discriminator=dict(depth=2),
    adaptive_generator_loss_weight_gain=0.75,
)
runner = dict(model=model)

_export_ = dict(trainer=runner, validator=runner)
