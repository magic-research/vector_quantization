__all__ = [
    'LlamaTransformer',
]

import todd
from todd.bases.registries import Item
from torch import nn
from transformers import LlamaConfig, LlamaForCausalLM

from vq.tasks.sequence_modeling.models import VQSMTransformerRegistry

from .hf import HFTransformer


@VQSMTransformerRegistry.register_()
class LlamaTransformer(HFTransformer):
    _transformer: LlamaForCausalLM

    @classmethod
    def transformer_build_pre_hook(
        cls,
        config: todd.Config,
        registry: todd.RegistryMeta,
        item: Item,
    ) -> todd.Config:
        llama_config = LlamaConfig(**config.transformer)
        config.transformer = LlamaForCausalLM(llama_config)
        return config

    def init_weights(self, config: todd.Config) -> bool:

        def initializer(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=0.02)

        super().init_weights(config)
        self._transformer.apply(initializer)
        nn.init.constant_(self._transformer.lm_head.weight, 0.0)
        return False
