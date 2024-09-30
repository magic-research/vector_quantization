from typing import Any

_kwargs_: dict[str, Any]
_kwargs_ = dict(_kwargs_)
image_size = _kwargs_['image_size']

transforms = [
    dict(type='Resize', size=image_size, interpolation=3),
    dict(type='RandomCrop', size=image_size),
    dict(type='RandomHorizontalFlip'),
    dict(type='PILToTensor'),
]

_export_ = dict(transforms=transforms)
