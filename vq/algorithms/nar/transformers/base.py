__all__ = [
    'BaseTransformer',
]

import enum
from abc import abstractmethod
from typing import TypeVar

import todd.tasks.large_multimodal_model as lmm
import torch
from todd.runners import Memo

from vq.tasks.sequence_modeling.models import BaseTransformer as BaseSMTransformer

import torch
import os
import math
import argparse
import models_mage
import numpy as np
from tqdm import tqdm
import cv2

T = TypeVar('T', bound=enum.Enum)


def mask_by_random_topk(mask_len, probs, temperature=1.0):
    mask_len = mask_len.squeeze()
    confidence = torch.log(probs) + torch.Tensor(
        temperature * np.random.gumbel(size=probs.shape)
    ).cuda()
    sorted_confidence, _ = torch.sort(confidence, axis=-1)
    # Obtains cut off threshold given the mask lengths.
    cut_off = sorted_confidence[:, mask_len.long() - 1:mask_len.long()]
    # Masks tokens with lower confidence.
    masking = (confidence <= cut_off)
    return masking


def gen_image(model, bsz, seed, num_iter=12, choice_temperature=4.5):
    torch.manual_seed(seed)
    np.random.seed(seed)
    codebook_emb_dim = 256
    codebook_size = 1024
    mask_token_id = model.mask_token_label
    unknown_number_in_the_beginning = 256
    _CONFIDENCE_OF_KNOWN_TOKENS = +np.inf

    initial_token_indices = mask_token_id * torch.ones(
        bsz, unknown_number_in_the_beginning
    )

    token_indices = initial_token_indices.cuda()

    for step in range(num_iter):
        cur_ids = token_indices.clone().long()

        token_indices = torch.cat([
            torch.zeros(token_indices.size(0),
                        1).cuda(device=token_indices.device), token_indices
        ],
                                  dim=1)
        token_indices[:, 0] = model.fake_class_label
        token_indices = token_indices.long()
        token_all_mask = token_indices == mask_token_id

        token_drop_mask = torch.zeros_like(token_indices)

        # token embedding
        input_embeddings = model.token_emb(token_indices)

        # encoder
        x = input_embeddings
        for blk in model.blocks:
            x = blk(x)
        x = model.norm(x)

        # decoder
        logits = model.forward_decoder(x, token_drop_mask, token_all_mask)
        logits = logits[:, 1:, :codebook_size]

        # get token prediction
        sample_dist = torch.distributions.categorical.Categorical(
            logits=logits
        )
        sampled_ids = sample_dist.sample()

        # get ids for next step
        unknown_map = (cur_ids == mask_token_id)
        sampled_ids = torch.where(unknown_map, sampled_ids, cur_ids)
        # Defines the mask ratio for the next round. The number to mask out is
        # determined by mask_ratio * unknown_number_in_the_beginning.
        ratio = 1. * (step + 1) / num_iter

        mask_ratio = np.cos(math.pi / 2. * ratio)

        # sample ids according to prediction confidence
        probs = torch.nn.functional.softmax(logits, dim=-1)
        selected_probs = torch.squeeze(
            torch.gather(
                probs, dim=-1, index=torch.unsqueeze(sampled_ids, -1)
            ), -1
        )

        selected_probs = torch.where(
            unknown_map, selected_probs.double(), _CONFIDENCE_OF_KNOWN_TOKENS
        ).float()

        mask_len = torch.Tensor([
            np.floor(unknown_number_in_the_beginning * mask_ratio)
        ]).cuda()
        # Keeps at least one of prediction in this round and also masks out at least
        # one and for the next iteration
        mask_len = torch.maximum(
            torch.Tensor([1]).cuda(),
            torch.minimum(
                torch.sum(unknown_map, dim=-1, keepdims=True) - 1, mask_len
            )
        )

        # Sample masking tokens for next iteration
        masking = mask_by_random_topk(
            mask_len[0], selected_probs, choice_temperature * (1 - ratio)
        )
        # Masks tokens with lower confidence.
        token_indices = torch.where(masking, mask_token_id, sampled_ids)

    # vqgan visualization
    z_q = model.vqgan.quantize.get_codebook_entry(
        sampled_ids, shape=(bsz, 16, 16, codebook_emb_dim)
    )
    gen_images = model.vqgan.decode(z_q)
    return gen_images


parser = argparse.ArgumentParser('MAGE generation', add_help=False)
parser.add_argument(
    '--temp', default=4.5, type=float, help='sampling temperature'
)
parser.add_argument(
    '--num_iter',
    default=12,
    type=int,
    help='number of iterations for generation'
)

args = parser.parse_args()

gen_images_batch = gen_image(
    choice_temperature=args.temp, num_iter=args.num_iter
)
