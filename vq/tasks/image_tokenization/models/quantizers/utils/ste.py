__all__ = [
    'ste',
]

import torch
import torch.distributed


def ste(z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return x + (z - x).detach()
