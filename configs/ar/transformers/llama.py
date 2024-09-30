from typing import Any

_kwargs_: dict[str, Any]
_kwargs_ = dict(_kwargs_)
transformer_size = _kwargs_['transformer_size']

TRANSFORMERS = dict(
    medium=dict(
        num_hidden_layers=24,
        num_attention_heads=16,
        hidden_size=1024,
        intermediate_size=2816,
        rms_norm_eps=1e-5,
    ),
)

_export_ = dict(
    type='LlamaTransformer',
    transformer=TRANSFORMERS[transformer_size],
)
