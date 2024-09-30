__all__ = [
    'VQGANTrainer',
]

from typing import Any, Mapping, TypeVar, cast

import todd
import torch
import torch.utils.data
from todd.bases.registries import Item, RegistryMeta
from todd.runners import CallbackRegistry, Memo
from todd.runners.callbacks import LRScheduleCallback, OptimizeCallback
from torch import nn

from vq import VQRunnerRegistry
from vq.datasets import Batch
from vq.runners import BaseTrainer

T = TypeVar('T', bound=nn.Module)


@VQRunnerRegistry.register_()
class VQGANTrainer(BaseTrainer[T]):

    def __init__(
        self,
        *args,
        optimizers: tuple[torch.optim.Optimizer, torch.optim.Optimizer],
        generator_optimize_callback: OptimizeCallback,
        generator_lr_schedule_callback: LRScheduleCallback | None = None,
        generator_start: int = 0,
        discriminator_start: int = 0,
        **kwargs,
    ) -> None:
        g_optimizer, d_optimizer = optimizers
        self._g_optimizer = g_optimizer
        self._d_optimizer = d_optimizer

        self._g_optimize_callback = generator_optimize_callback
        self._g_lr_schedule_callback = generator_lr_schedule_callback

        assert min(generator_start, discriminator_start) == 0
        self._g_start = generator_start
        self._d_start = discriminator_start

        super().__init__(*args, optimizer=d_optimizer, **kwargs)

    @classmethod
    def optimizer_build_pre_hook(
        cls,
        config: todd.Config,
        registry: RegistryMeta,
        item: Item,
    ) -> todd.Config:
        assert 'optimizer' not in config

        config.optimizer = config.optimizers.generator
        config = super().optimizer_build_pre_hook(config, registry, item)
        g_optimizer = config.pop('optimizer')

        config.optimizer = config.optimizers.discriminator
        config = super().optimizer_build_pre_hook(config, registry, item)
        d_optimizer = config.pop('optimizer')

        config.optimizers = (g_optimizer, d_optimizer)
        return config

    @classmethod
    def callbacks_build_pre_hook(
        cls,
        config: todd.Config,
        registry: RegistryMeta,
        item: Item,
    ) -> todd.Config:
        config = super().callbacks_build_pre_hook(config, registry, item)
        config.generator_optimize_callback = CallbackRegistry.build_or_return(
            config.generator_optimize_callback,
        )
        if (
            generator_lr_schedule_callback :=
            config.get('generator_lr_schedule_callback')
        ) is not None:
            config.generator_lr_schedule_callback = \
                CallbackRegistry.build_or_return(
                    generator_lr_schedule_callback,
                )
        return config

    @property
    def with_g(self) -> bool:
        return self._iter > self._g_start

    @property
    def with_d(self) -> bool:
        return self._iter > self._d_start

    def _init_callbacks(self, *args, **kwargs) -> None:
        super()._init_callbacks(*args, **kwargs)
        self._g_optimize_callback.bind(self)
        if self._g_lr_schedule_callback is not None:
            with todd.utils.set_temp(self, '._optimizer', self._g_optimizer):
                self._g_lr_schedule_callback.bind(self)

    def _choose_optimizer(self, optimizer: torch.optim.Optimizer) -> None:
        self._optimizer = optimizer
        self._strategy.module.requires_grad_(False)
        for param_group in optimizer.param_groups:
            for param in param_group['params']:
                cast(nn.Parameter, param).requires_grad_(True)

    def _run_iter(self, batch: Batch, memo: Memo, *args, **kwargs) -> Memo:
        if self.with_g:
            self._choose_optimizer(self._g_optimizer)
            memo = super()._run_iter(
                batch,
                memo,
                *args,
                mode='generation',
                **kwargs,
            )
            self._g_optimize_callback.after_run_iter(batch, memo)
            if self._g_lr_schedule_callback is not None:
                self._g_lr_schedule_callback.after_run_iter(batch, memo)
            if 'log' in memo:
                log: Memo = memo['log']
                log['g_grad'] = log.pop('grad', None)
                log['g_lr'] = log.pop('lr', None)
        if self.with_d:
            self._choose_optimizer(self._d_optimizer)
            memo = super()._run_iter(
                batch,
                memo,
                *args,
                mode='discrimination',
                **kwargs,
            )
            # optimize and lr_schedule callbacks are called automatically after
            # `_run_iter`
        else:
            # no parameter gets optimized
            self._strategy.module.requires_grad_(False)
            memo['loss'] = torch.zeros([], requires_grad=True)
        return memo

    def state_dict(self, *args, **kwargs) -> dict[str, Any]:
        self._choose_optimizer(self._d_optimizer)
        state_dict = super().state_dict(*args, **kwargs)
        self._choose_optimizer(self._g_optimizer)
        state_dict['g_optimizer'] = \
            self._strategy.optim_state_dict(*args, **kwargs)
        state_dict['g_optimize_callback'] = \
            self._g_optimize_callback.state_dict(*args, **kwargs)
        if self._g_lr_schedule_callback is not None:
            state_dict['g_lr_schedule_callback'] = \
                self._g_lr_schedule_callback.state_dict(*args, **kwargs)
        return state_dict

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        *args,
        **kwargs,
    ) -> None:
        self._choose_optimizer(self._d_optimizer)
        super().load_state_dict(state_dict, *args, **kwargs)
        self._choose_optimizer(self._g_optimizer)
        self._strategy.load_optim_state_dict(state_dict['g_optimizer'])
        self._g_optimize_callback.load_state_dict(
            state_dict['g_optimize_callback'],
        )
        if self._g_lr_schedule_callback is not None:
            self._g_lr_schedule_callback.load_state_dict(
                state_dict['g_lr_schedule_callback'],
            )
