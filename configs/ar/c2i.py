_base_ = [
    'x2i.py',
]

trainer = dict(model=dict(type='ARC2I'))
validator = dict(model=dict(type='ARC2I'))
