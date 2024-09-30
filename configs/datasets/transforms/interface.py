from typing import Any

from todd.configs import PyConfig

_kwargs_: dict[str, Any]
_kwargs_ = dict(_kwargs_)
image_size = _kwargs_['image_size']
augmentation = _kwargs_['augmentation']

config = PyConfig.load(
    f'configs/datasets/transforms/{augmentation}.py',
    image_size=image_size,
)
config.image_size = image_size
trainer = dict(dataset=config)

config = PyConfig.load(
    'configs/datasets/transforms/none.py',
    image_size=image_size,
)
config.image_size = image_size
validator = dict(dataset=config)

_export_ = dict(trainer=trainer, validator=validator)
