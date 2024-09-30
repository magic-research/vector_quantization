__all__ = [
    'BaseReconstructLoss',
    'L1Loss',
    'LPIPSLoss',
    'SSIMLoss',
    'PSNRLoss',
]

from abc import ABC, abstractmethod

import einops
import todd
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
from todd.bases.registries import BuildPreHookMixin, Item
from todd.models import losses
from todd.models.filters import NamedParametersFilter
from torch import nn
from torchvision import models

from vq.utils import Store

from .registries import VQIRLossRegistry


class BaseReconstructLoss(losses.BaseLoss, ABC):

    @abstractmethod
    def forward(  # pylint: disable=arguments-differ
        self,
        pred_image: torch.Tensor,
        image: torch.Tensor,
    ) -> torch.Tensor:
        pass


@VQIRLossRegistry.register_()
class L1Loss(BaseReconstructLoss, BuildPreHookMixin):

    def __init__(self, *args, l1: losses.L1Loss, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._l1 = l1

    @classmethod
    def build_pre_hook(
        cls,
        config: todd.Config,
        registry: todd.RegistryMeta,
        item: Item,
    ) -> todd.Config:
        config = super().build_pre_hook(config, registry, item)
        config.l1 = losses.L1Loss(
            reduction='none',
            **config.get_config('l1'),
        )
        return config

    def forward(
        self,
        pred_image: torch.Tensor,
        image: torch.Tensor,
    ) -> torch.Tensor:
        loss = self._l1(pred_image, image)
        return self._reduce(loss)


@VQIRLossRegistry.register_()
class MSELoss(BaseReconstructLoss, BuildPreHookMixin):

    def __init__(self, *args, mse: losses.MSELoss, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._mse = mse

    @classmethod
    def build_pre_hook(
        cls,
        config: todd.Config,
        registry: todd.RegistryMeta,
        item: Item,
    ) -> todd.Config:
        config = super().build_pre_hook(config, registry, item)
        config.mse = losses.MSELoss(
            reduction='none',
            **config.get_config('mse'),
        )
        return config

    def forward(
        self,
        pred_image: torch.Tensor,
        image: torch.Tensor,
    ) -> torch.Tensor:
        loss = self._mse(pred_image, image)
        return self._reduce(loss)


@VQIRLossRegistry.register_()
class LPIPSLoss(
    todd.models.MeanStdMixin,
    todd.models.NoGradMixin,
    BaseReconstructLoss,
):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(
            *args,
            mean=(-.030, -.088, -.188),
            std=(.458, .448, .450),
            no_grad=NamedParametersFilter(),
            **kwargs,
        )

        weights = models.VGG16_Weights.IMAGENET1K_V1
        vgg = models.vgg16(weights=weights)
        for i in [3, 8, 15, 22, 29]:
            module: nn.Module = vgg.features[i]
            module.register_forward_hook(self._forward_hook)
        self._vgg = vgg

        self._dropout = nn.Dropout()

        self._convs = nn.ModuleList(
            nn.Conv2d(in_channels, 1, 1, bias=False)
            for in_channels in [64, 128, 256, 512, 512]
        )

    def reset_features(self) -> None:
        self._features: list[torch.Tensor] = []

    def init_weights(self, config: todd.Config) -> bool:
        config.setdefault('pretrained', 'pretrained/lpips/vgg.pth.converted')
        state_dict = todd.patches.torch.load(
            config.pretrained,
            'cpu',
            directory=Store.PRETRAINED,
        )
        self._convs.load_state_dict(state_dict, strict=False)
        return False

    def _forward_hook(
        self,
        module: nn.Module,
        inputs: tuple[torch.Tensor],
        output: torch.Tensor,
    ) -> None:
        output = F.normalize(output, p=2, dim=1, eps=1e-10)
        self._features.append(output)

    def extract_features(self, image: torch.Tensor) -> list[torch.Tensor]:
        image = self.normalize(image)
        self.reset_features()
        self._vgg(image)
        outs = list(self._features)
        self.reset_features()
        return outs

    def forward(
        self,
        pred_image: torch.Tensor,
        image: torch.Tensor,
    ) -> torch.Tensor:
        pred_features = self.extract_features(pred_image)
        features = self.extract_features(image)
        losses_ = pred_image.new_zeros([])
        for pred_feature, feature, conv in zip(
            pred_features,
            features,
            self._convs,
        ):
            loss = F.mse_loss(pred_feature, feature, reduction='none')
            loss = self._dropout(loss)
            loss = conv(loss)
            # TODO: change to (b, c)
            loss = einops.reduce(loss, 'b 1 h w -> b 1 1 1', 'mean')
            losses_ = losses_ + loss
        return self._reduce(losses_)


@VQIRLossRegistry.register_()
class SSIMLoss(BaseReconstructLoss):

    def forward(
        self,
        pred_image: torch.Tensor,
        image: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate the Structural Similarity Index Measure (SSIM).

        Args:
            pred_image: The predicted images. This should be a 4D tensor with
                shape (batch_size, num_channels, height, width). The values
                should be between 0 and 1 (i.e., they should be normalized).
            image: The target images. This should have the same shape and
                value range as the prediction tensor.

        Returns:
            The SSIM between the predicted and target images.
        """
        pred_image_np = pred_image.cpu().numpy()
        image_np = image.cpu().numpy()
        ssim_list = [
            ssim(pi, i, channel_axis=0, data_range=1)
            for pi, i in zip(pred_image_np, image_np)
        ]
        ssim_tensor = pred_image.new_tensor(ssim_list)
        return self._reduce(ssim_tensor)


@VQIRLossRegistry.register_()
class PSNRLoss(MSELoss):

    def forward(
        self,
        pred_image: torch.Tensor,
        image: torch.Tensor,
    ) -> torch.Tensor:
        loss: torch.Tensor = self._mse(pred_image, image)
        loss = einops.reduce(loss, 'b ... -> b', 'mean')
        loss = -10 * loss.log10()
        return self._reduce(loss)
