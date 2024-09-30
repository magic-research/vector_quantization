from typing import Any

_kwargs_: dict[str, Any]
_kwargs_ = dict(_kwargs_)
image_size = _kwargs_['image_size']

transforms = [
    dict(
        type='RandomResizedCrop',
        size=image_size,
        interpolation=3,
        scale=(0.8, 1.0),
    ),
    dict(type='RandomHorizontalFlip'),
    dict(type='PILToTensor'),
]

_export_ = dict(transforms=transforms)
