from typing import Any

_kwargs_: dict[str, Any]
_kwargs_ = dict(_kwargs_)
ir_config = _kwargs_['ir_config']
it_state_dict = _kwargs_.get('it_state_dict')
ir_state_dict = _kwargs_['ir_state_dict']

_base_ = [
    'transformers/interface.py',
]

state_dicts = ([ir_state_dict]
               if it_state_dict is None else [ir_state_dict, it_state_dict])
model = dict(
    type='ARC2I',
    freeze=dict(type='NamedModulesFilter', name='_ir'),
    filter_state_dict=True,
    transformer=dict(sampler=dict(type='TopKTopPSampler')),
    ir=dict(config=ir_config, state_dicts=state_dicts, strict=False),
)
runner = dict(model=model)

_export_ = dict(trainer=runner, validator=runner)
