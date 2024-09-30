from typing import Any

_kwargs_: dict[str, Any]
_kwargs_ = dict(_kwargs_)
it_config = _kwargs_['it_config']
it_state_dict = _kwargs_['it_state_dict']

model = dict(
    type='VQModelRegistry.VQICModelRegistry.BaseModel',
    freeze=dict(type='NamedModulesFilter', name='_it'),
    filter_state_dict=True,
    it=dict(config=it_config, state_dicts=[it_state_dict], strict=False),
)
runner = dict(model=model)

_export_ = dict(trainer=runner, validator=runner)
