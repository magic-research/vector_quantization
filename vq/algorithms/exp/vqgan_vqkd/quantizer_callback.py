__all__ = [
    'VQGAN_VQKDCallback',
]

import random

import einops
import todd
import torch
import torch.distributed
import torch.nn.functional as F
import tqdm
from todd.patches.torch import get_rank, get_world_size
from todd.runners import Memo

from vq.algorithms.vq.callbacks import UpdateMixin
from vq.algorithms.vq.callbacks.normalize import NormalizeCallback
from vq.algorithms.vq.utils import QuantStatistics
from vq.tasks.image_tokenization.models.quantizers import (
    VQITQuantizerCallbackRegistry,
)
from vq.tasks.image_tokenization.models.quantizers.callbacks import (
    LazyInitWeightsMixin,
)


def distributed_cat(x: torch.Tensor) -> torch.Tensor:
    if get_world_size() <= 1:
        return x
    if get_rank() > 0:
        torch.distributed.gather(x)
        return x.new_empty(0)
    gather_list = [torch.zeros_like(x) for _ in range(get_world_size())]
    torch.distributed.gather(x, gather_list)
    x = torch.cat(gather_list)
    return x


@VQITQuantizerCallbackRegistry.register_()
class VQGAN_VQKDCallback(  # noqa: N801 pylint: disable=invalid-name
    LazyInitWeightsMixin,
    UpdateMixin,
    NormalizeCallback,
):

    def _kmeans(
        self,
        x: torch.Tensor,
        quant: torch.Tensor,
        sync: bool,
    ) -> torch.Tensor:
        e = self.vector_quantizer.embeddings

        quant_statistics = QuantStatistics(
            quant=quant,
            codebook_size=e.shape[0],
            sync=sync,
        )
        occurrences = quant_statistics.bin_count()
        occurrences = einops.rearrange(occurrences, 'e -> e 1')

        quant = einops.repeat(quant, 'x -> x d', d=e.shape[1])
        centroids = torch.zeros_like(e)
        centroids.scatter_add_(0, quant, x)
        if sync and get_world_size() > 1:
            torch.distributed.all_reduce(centroids)

        occurred = occurrences > 0
        occurrences.clamp_min_(1)

        centroids = centroids / occurrences
        centroids = centroids.where(occurred, e)
        return centroids

    def _update_embedding(self, e: torch.Tensor) -> None:
        e = F.normalize(e)
        return super()._update_embedding(e)

    def lazy_init_weights(
        self,
        config: todd.Config,
        x: torch.Tensor,
        memo: Memo,
    ) -> None:
        super().lazy_init_weights(config, x, memo)
        if not self.quantizer.training:
            return

        x = distributed_cat(x)
        e = self.vector_quantizer.embeddings
        iters = config.get('iters', 10)

        if get_rank() > 0:
            e = torch.empty_like(e)
        elif x.shape[0] < e.shape[0]:
            e[:x.shape[0]] = x
        else:
            x = F.normalize(x)
            if offload := x.shape[0] * e.shape[0] > 2**30:
                todd.logger.info("Offloading to CPU")
                device = x.device
                x = x.cpu()
            indices = random.sample(range(x.shape[0]), e.shape[0])
            e = x[indices]
            for _ in tqdm.trange(iters, desc='K-Means', leave=False):
                self._update_embedding(e)
                quant, memo = self.vector_quantizer._encode(x, memo)
                e = self._kmeans(x, quant, False)
            if offload:
                e = e.to(device)
        if get_world_size() > 1:
            torch.distributed.broadcast(e, 0)

        self._update_embedding(e)

    def after_encode(
        self,
        x: torch.Tensor,
        quant: torch.Tensor,
        memo: Memo,
    ) -> torch.Tensor:
        quant = super().after_encode(x, quant, memo)
        if not self.quantizer.training:
            return quant

        e = self.vector_quantizer.embeddings
        e = F.normalize(e)
        e = self._ema(self.vector_quantizer.embeddings, e)
        self._update_embedding(e)
        return quant
