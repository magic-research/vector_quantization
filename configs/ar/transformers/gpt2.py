from typing import Any

_kwargs_: dict[str, Any]
_kwargs_ = dict(_kwargs_)
transformer_size = _kwargs_['transformer_size']

_export_ = dict(
    type='GPT2Transformer',
    transformer=f'pretrained/huggingface/gpt2-{transformer_size}',
)
