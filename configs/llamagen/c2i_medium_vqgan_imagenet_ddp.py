"""C2I Medium with VQGAN tokenizer.

Example:
    bash tools/torchrun.sh -m vq.train llamagen/c2i_medium_vqgan_imagenet_ddp \
configs/llamagen/c2i_medium_vqgan_imagenet_ddp.py --config-options \
ir_state_dict::work_dirs/llamagen/vqgan_imagenet_ddp/checkpoints/iter_1/model.\
pth
"""

from typing import Any

from todd.configs import PyConfig

_kwargs_: dict[str, Any]
_kwargs_ = dict(_kwargs_)
_kwargs_.setdefault('ir_config', 'configs/llamagen/vqgan_imagenet_ddp.py')

_export_ = PyConfig.load(
    'configs/llamagen/c2i_medium_imagenet_ddp.py',
    **_kwargs_,
)
