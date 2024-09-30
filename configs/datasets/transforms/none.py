from typing import Any

_kwargs_: dict[str, Any]
_kwargs_ = dict(_kwargs_)
image_size = _kwargs_['image_size']

transforms = [
    dict(type='Resize', size=image_size, interpolation=3),
    dict(type='CenterCrop', size=image_size),
    dict(type='PILToTensor'),
]

_export_ = dict(transforms=transforms)
