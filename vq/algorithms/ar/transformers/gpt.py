__all__ = [
    'GPT2Transformer',
]

import os

import todd
from todd.bases.registries import Item
from transformers import GPT2LMHeadModel

from vq.tasks.sequence_modeling.models import VQSMTransformerRegistry
from vq.utils import Store

from .hf import HFTransformer


@VQSMTransformerRegistry.register_()
class GPT2Transformer(HFTransformer):
    _transformer: GPT2LMHeadModel

    @classmethod
    def transformer_build_pre_hook(
        cls,
        config: todd.Config,
        registry: todd.RegistryMeta,
        item: Item,
    ) -> todd.Config:
        transformer = (
            'pretrained/huggingface/gpt2'
            if todd.Store.DRY_RUN else config.transformer
        )
        transformer = os.path.join(Store.PRETRAINED, transformer)
        config.transformer = GPT2LMHeadModel.from_pretrained(transformer)
        return config
