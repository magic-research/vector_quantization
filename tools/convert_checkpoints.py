import argparse
import importlib
import pathlib
import re
import sys
import unittest.mock
from abc import ABC, abstractmethod

import einops
import todd
import torch
from todd.configs import PyConfig
from todd.patches.py import DictAction
from todd.registries import ModelRegistry
from torch import nn

StateDict = dict[str, torch.Tensor]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Convert Checkpoints')
    parser.add_argument('--options', type=DictAction, default=dict())
    parser.add_argument('--check')
    parser.add_argument('--suffix', default='.converted')

    sub_parsers = parser.add_subparsers()
    for k, v in ConverterRegistry.items():
        sub_parser = sub_parsers.add_parser(k)
        sub_parser.add_argument('path', type=pathlib.Path)
        sub_parser.set_defaults(type=v)
    args = parser.parse_args()
    return args


class ConverterRegistry(todd.Registry):
    pass


class BaseConverter(ABC):

    def load(self, path: pathlib.Path) -> StateDict:
        return torch.load(path, 'cpu')

    @abstractmethod
    def convert(self, key: str) -> str | None:
        pass

    def fix_incompatible_keys(self, state_dict: StateDict) -> StateDict:
        return state_dict


@ConverterRegistry.register_('lpips')
class LPIPSConverter(BaseConverter):

    def convert(self, key: str) -> str | None:
        pattern = re.compile('^lin([01234]).model.1.weight$')
        match = pattern.match(key)
        if match:
            return f'{match.group(1)}.weight'
        return None


@ConverterRegistry.register_('taming')
class TamingTransformersConverter(BaseConverter):

    def _merge_attention(self, state_dict: StateDict, prefix: str) -> None:
        norm_keys = ['norm.weight', 'norm.bias']
        norms = {k: state_dict.pop(prefix + k) for k in norm_keys}

        weight_keys = ['q.weight', 'k.weight', 'v.weight', 'proj_out.weight']
        weights = [state_dict.pop(prefix + k) for k in weight_keys]
        weights = [einops.rearrange(w, 'oc ic 1 1 -> oc ic') for w in weights]

        bias_keys = ['q.bias', 'k.bias', 'v.bias', 'proj_out.bias']
        biases = [state_dict.pop(prefix + k) for k in bias_keys]

        state_dict.update((f'{prefix}_group_{k}', v) for k, v in norms.items())

        mha_prefix = prefix + '_multihead_attention.'
        state_dict[f'{mha_prefix}in_proj_weight'] = torch.cat(weights[:-1])
        state_dict[f'{mha_prefix}out_proj.weight'] = weights[-1]
        state_dict[f'{mha_prefix}in_proj_bias'] = torch.cat(biases[:-1])
        state_dict[f'{mha_prefix}out_proj.bias'] = biases[-1]

    def _merge_attentions(self, state_dict: StateDict) -> None:
        for prefix in [
            'encoder.down.4.attn.0.',
            'encoder.down.4.attn.1.',
            'encoder.mid.attn_1.',
            'decoder.up.4.attn.0.',
            'decoder.up.4.attn.1.',
            'decoder.up.4.attn.2.',
            'decoder.mid.attn_1.',
        ]:
            self._merge_attention(state_dict, prefix)

    def load(self, path: pathlib.Path) -> StateDict:
        # Mock pytorch_lightning to load the checkpoint
        sys.modules['pytorch_lightning'] = unittest.mock.Mock()
        sys.modules['pytorch_lightning.callbacks.model_checkpoint'] = \
            unittest.mock.Mock()
        checkpoint = torch.load(path, 'cpu')
        state_dict = checkpoint['state_dict']
        self._merge_attentions(state_dict)
        return state_dict

    def _convert_residual(self, key: str) -> str:
        if key.startswith('norm1.'):
            return '_residual.0.' + key.removeprefix('norm1.')
        if key.startswith('conv1.'):
            return '_residual.2.' + key.removeprefix('conv1.')
        if key.startswith('norm2.'):
            return '_residual.3.' + key.removeprefix('norm2.')
        if key.startswith('conv2.'):
            return '_residual.5.' + key.removeprefix('conv2.')
        if key.startswith('nin_shortcut.'):
            return '_shortcut.' + key.removeprefix('nin_shortcut.')
        raise ValueError(key)

    def _convert_residuals(self, key: str) -> str:
        index = key.index('.') + 1
        prefix = key[:index]
        key = key[index:]
        return '_residuals.' + prefix + self._convert_residual(key)

    def _convert_resample(self, key: str) -> str:
        if key.startswith('conv.'):
            key = key.removeprefix('conv.')
            return '_conv.' + key
        raise ValueError(key)

    def _convert_attentions(self, key: str) -> str:
        index = key.index('.') + 1
        prefix = key[:index]
        key = key[index:]
        return '_attentions.' + prefix + key

    def _convert_layer(self, key: str) -> str:
        if key.startswith('block.'):
            key = key.removeprefix('block.')
            return self._convert_residuals(key)
        if key.startswith('attn.'):
            key = key.removeprefix('attn.')
            return self._convert_attentions(key)
        raise ValueError(key)

    def _convert_encoder_down(self, key: str) -> str:
        index = key.index('.') + 1
        prefix = key[:index]
        key = key[index:]
        if key.startswith('downsample.'):
            key = key.removeprefix('downsample.')
            key = self._convert_resample(key)
            return '_resamples.' + prefix + key
        return '_layers.' + prefix + self._convert_layer(key)

    def _convert_mid(self, key: str) -> str:
        if key.startswith('block_1.'):
            key = key.removeprefix('block_1.')
            key = '0.' + self._convert_residual(key)
        elif key.startswith('attn_1.'):
            key = '1.' + key.removeprefix('attn_1.')
        elif key.startswith('block_2.'):
            key = key.removeprefix('block_2.')
            key = '2.' + self._convert_residual(key)
        else:
            raise ValueError(key)
        return '_refine.' + key

    def _convert_encoder(self, key: str) -> str:
        if key.startswith('down.'):
            key = key.removeprefix('down.')
            key = self._convert_encoder_down(key)
        elif key.startswith('mid.'):
            key = key.removeprefix('mid.')
            key = self._convert_mid(key)
        elif key.startswith('conv_in.'):
            key = '_in_conv.' + key.removeprefix('conv_in.')
        elif key.startswith('norm_out.'):
            key = '_projector.0.' + key.removeprefix('norm_out.')
        elif key.startswith('conv_out'):
            key = '_projector.2.' + key.removeprefix('conv_out.')
        else:
            raise ValueError(key)
        return '_encoder.' + key

    def _convert_decoder_up(self, key: str) -> str:
        index = key.index('.')
        prefix = key[:index]
        key = key[index + 1:]
        if key.startswith('upsample.'):
            key = key.removeprefix('upsample.')
            key = self._convert_resample(key)
            return f'_resamples.{4 - int(prefix)}.' + key
        return f'_layers.{4 - int(prefix)}.' + self._convert_layer(key)

    def _convert_decoder(self, key: str) -> str:
        if key.startswith('up.'):
            key = key.removeprefix('up.')
            key = self._convert_decoder_up(key)
        elif key.startswith('mid.'):
            key = key.removeprefix('mid.')
            key = self._convert_mid(key)
        elif key.startswith('conv_in.'):
            key = '_in_conv.' + key.removeprefix('conv_in.')
        elif key.startswith('norm_out.'):
            key = '_projector.0.' + key.removeprefix('norm_out.')
        elif key.startswith('conv_out'):
            key = '_projector.2.' + key.removeprefix('conv_out.')
        else:
            raise ValueError(key)
        return '_decoder.' + key

    def _convert_loss(self, key: str) -> str:
        if key.startswith('perceptual_loss.'):
            key = key.removeprefix('perceptual_loss.')
            return '_reconstruct_losses.lpips_r_loss._lpips.' + key
        if key.startswith('discriminator.main.'):
            key = key.removeprefix('discriminator.main.')
            return '_discriminator.' + key
        raise ValueError(key)

    def _convert_quantizer(self, key: str) -> str:
        if key.startswith('embedding.'):
            key = key.removeprefix('embedding.')
            key = '_embedding.' + key
        return '_quantizer.' + key

    def convert(self, key: str) -> str | None:
        if key.startswith('encoder.'):
            key = key.removeprefix('encoder.')
            return self._convert_encoder(key)
        if key.startswith('quant_conv.'):
            key = key.removeprefix('quant_conv.')
            return '_post_encode._conv.' + key
        if key.startswith('quantize.'):
            key = key.removeprefix('quantize.')
            return self._convert_quantizer(key)
        if key.startswith('post_quant_conv.'):
            key = key.removeprefix('post_quant_conv.')
            return '_pre_decode._conv.' + key
        if key.startswith('decoder.'):
            key = key.removeprefix('decoder.')
            return self._convert_decoder(key)
        if key.startswith('loss.'):
            key = key.removeprefix('loss.')
            return self._convert_loss(key)
        return key

    def fix_incompatible_keys(self, state_dict: StateDict) -> StateDict:
        tensor = torch.tensor(1)
        state_dict.update({
            '_quant_loss._weight._steps': tensor,
            '_reconstruct_losses.l1_reconstruct_loss._weight._steps': tensor,
            '_reconstruct_losses.lpips_r_loss._weight._steps': tensor,
            '_generator_loss._weight._steps': tensor,
            '_discriminator_loss._weight._steps': tensor,
        })
        return state_dict


@ConverterRegistry.register_('beitv2')
class BEiTv2Converter(BaseConverter):

    def __init__(self, with_decoder: bool) -> None:
        self._with_decoder = with_decoder

    def _convert_quantizer(self, key: str) -> str:
        if key.startswith('embedding.'):
            key = '_' + key
        elif key == 'embed_prob':
            key = '_probability'
        return '_quantizer.' + key

    def _convert_encoder_task_layer(self, key: str) -> str:
        return '_encoder.task_layer.' + key

    def _convert_decoder_task_layer(self, key: str) -> str:
        if not self._with_decoder:
            return ''
        return '_decoder.task_layer.' + key

    def convert(self, key: str) -> str | None:
        if key.startswith('encoder.'):
            return '_' + key
        if key.startswith('encode_task_layer.'):
            key = key.removeprefix('encode_task_layer.')
            return self._convert_encoder_task_layer(key)
        if key.startswith('post_quant_conv.'):
            key = key.removeprefix('post_quant_conv.')
            return '_pre_decode._conv.' + key
        if key.startswith('quantize.'):
            key = key.removeprefix('quantize.')
            return self._convert_quantizer(key)
        if key.startswith('decode_task_layer.'):
            key = key.removeprefix('decode_task_layer.')
            return self._convert_decoder_task_layer(key)
        if key.startswith('decoder_norm.'):
            if self._with_decoder:
                return '_decoder.' + key
            return None
        if key.startswith('decoder.'):
            if self._with_decoder:
                return '_' + key
            return None
        if key.startswith('scaling_layer.'):
            return None
        if key.startswith('teacher_model.'):
            return None
        raise ValueError(f"Unknown key: {key}")

    def fix_incompatible_keys(self, state_dict: StateDict) -> StateDict:
        tensor = torch.tensor(1)
        state_dict.update({
            '_quantizer._losses.mse_q_loss._weight._steps': tensor,
            '_quantizer._losses.mse_q_loss._mse._weight._steps': tensor,
            '_distiller._losses.0.0._weight._steps': tensor,
            '_distiller._losses.0.0._cosine_embedding._weight._steps': tensor,
        })
        return state_dict


def main() -> None:
    args = parse_args()

    converter: BaseConverter = args.type(**args.options)

    state_dict = converter.load(args.path)

    converted_state_dict: StateDict = dict()
    for k, v in state_dict.items():
        if (converted_k := converter.convert(k)) is not None:
            converted_state_dict[converted_k] = v

    converted_state_dict = converter.fix_incompatible_keys(
        converted_state_dict,
    )

    if args.check is not None:
        config = PyConfig.load(args.check)
        for custom_import in config.custom_imports:
            importlib.import_module(custom_import)
        config = config.trainer.strategy.model
        model: nn.Module = ModelRegistry.build(config)
        model.load_state_dict(state_dict)

    torch.save(converted_state_dict, str(args.path) + args.suffix)


if __name__ == '__main__':
    main()
