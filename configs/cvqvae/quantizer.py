callback = dict(
    type='CVQVAECallback',
    ema=dict(),
    anchor=dict(type='NearestAnchor'),
)
_export_ = dict(trainer=dict(model=dict(quantizer=dict(callbacks=[callback]))))
