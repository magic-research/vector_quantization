import math
from typing import cast
import torch
from todd.runners import Memo
from vq.tasks.sequence_modeling.models import BaseSampler, VQSMSamplerRegistry
from vq.tasks.sequence_modeling.models import VQSMTransformerRegistry
from ...utils.misc import get_memo
from .transformers import BaseTransformer
import todd.tasks.large_multimodal_model as lmm


@VQSMSamplerRegistry.register_()
class MAGESampler(BaseSampler):

    def __init__(
        self,
        *args,
        iters: int = 12,
        temperature: float = 4.5,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._iters = iters
        self._temperature = temperature
        self._gumbel = torch.distributions.Gumbel(0., 1.)

    @property
    def iters(self) -> int:
        return self._iters

    def sample(
        self,
        logits: torch.Tensor,
        memo: Memo,
    ) -> tuple[torch.Tensor, Memo]:
        tokens: torch.Tensor = memo['tokens']
        known: torch.Tensor = memo['known']
        iter_: int = memo.get('iter', 1)

        unknown = ~known
        assert unknown.any()
        num_unknowns, = set(unknown.sum(1).tolist())

        progress = iter_ / self._iters
        remaining_unknown_ratio = math.cos(math.pi / 2. * progress)
        num_remaining_unknowns = int(num_unknowns * remaining_unknown_ratio)
        memo.update(iter_=iter_ + 1, num_masks=num_remaining_unknowns)

        sampled_tokens, memo = super().sample(logits, memo)

        if num_remaining_unknowns == 0:
            known = torch.ones_like(known)
        else:
            temperature = self._temperature * (1 - progress)
            gumbel: torch.Tensor = self._gumbel.sample(tokens.shape)
            gumbel = gumbel.to(tokens.device)
            token_logits = logits[sampled_tokens] + temperature * gumbel
            token_logits[known] = float('inf')
            _, remaining_unknown_indices = token_logits.topk(
                num_remaining_unknowns,
                largest=False,
            )
            known = torch.ones_like(known)
            known[remaining_unknown_indices] = False

        sampled_tokens = torch.where(known, tokens, sampled_tokens)
        memo.update(tokens=sampled_tokens, known=known)
        return sampled_tokens, memo


@VQSMTransformerRegistry.register_()
class MAGETransformer(BaseTransformer):
    _sampler: MAGESampler

    @torch.no_grad()
    def inference(
        self,
        tokens: torch.Tensor,
        memo: Memo,
    ) -> tuple[torch.Tensor, Memo]:
        ...

    def _generate(
        self,
        tokens: torch.Tensor,
        length: int,
        codebook: lmm.Codebook[T],
        memo: Memo,
    ) -> tuple[torch.Tensor, Memo]:
        assert tokens.shape[1] < length
        mask_tokens = tokens.new_full(
            (tokens.shape[0], length - tokens.shape[1]),
            codebook.end - 1,
        )
        tokens = torch.cat((tokens, mask_tokens), 1)

        known = torch.zeros_like(tokens, dtype=torch.bool)
        known[:, :tokens.shape[1]] = True

        sampler_memo = get_memo(memo, 'sampler')
        sampler_memo.update(tokens=tokens, known=known)

        while True:
            logits, memo = self.inference(tokens, memo)
            tokens, memo = self.sample(logits, codebook, memo)
            known = cast(torch.Tensor, memo['sampler']['known']
            if known.all():
                break
        assert tokens.shape[1] == length
        return tokens, memo
