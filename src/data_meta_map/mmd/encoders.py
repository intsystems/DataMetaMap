"""Pluggable representation backends for the MMD embedder.

Three concrete strategies are provided:

* :class:`RawEncoder` - flatten samples to ``[N, D]`` (no learning).
* :class:`PretrainedEncoder` - run samples through a frozen
  feature extractor obtained via
  :func:`data_meta_map.models.get_model` (e.g. an ImageNet ResNet).
* :class:`GenerativeEncoder` - train a small autoencoder on the
  dataset and use the bottleneck activations as features.

All encoders expose the same ``transform(X) -> torch.Tensor`` interface
so :class:`data_meta_map.mmd_embedder.MMDEmbedder` can swap them in
freely.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(ABC):
    """Abstract feature extractor mapping a flat tensor ``X`` to features."""

    @abstractmethod
    def transform(self, X: torch.Tensor) -> torch.Tensor:
        """Return the encoded features of ``X``."""

    def fit(self, X: torch.Tensor) -> "Encoder":
        """Optionally fit the encoder on a sample. Default: no-op."""
        return self


class RawEncoder(Encoder):
    """Identity encoder: returns the input features unchanged.

    The MMD embedder always preprocesses inputs into a flat ``[N, D]``
    tensor first, so this encoder is effectively a passthrough that
    runs MMD directly on the raw pixel / feature space.
    """

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        return X


class PretrainedEncoder(Encoder):
    """Frozen pretrained feature extractor (e.g. ImageNet ResNet).

    Input tensors are reshaped to ``[N, channels, H, W]`` before being
    passed through ``model``; everything except the final classifier is
    treated as the feature extractor (identical to the recipe used in
    :mod:`data_meta_map.benchmarks.pretrain_benchmark.get_pretrained_to_task`).

    Args:
        model_name: Name resolvable by
            :func:`data_meta_map.models.get_model` (``"resnet18"`` or
            ``"resnet34"``). If ``model`` is supplied, this is ignored.
        pretrained: Whether to download pretrained ImageNet weights.
        model: Optional pre-built ``nn.Module``. If provided, takes
            precedence over ``model_name``.
        image_shape: ``(C, H, W)`` shape used to reshape flat inputs.
            Defaults to ``(3, 224, 224)`` for the bundled ResNets.
        batch_size: Batch size for the encoder pass.
        device: Torch device.
    """

    def __init__(
        self,
        model_name: str = "resnet18",
        *,
        pretrained: bool = True,
        model: Optional[nn.Module] = None,
        image_shape: tuple = (3, 224, 224),
        batch_size: int = 64,
        device: Union[str, torch.device] = "cpu",
    ):
        self.model_name = model_name
        self.image_shape = tuple(image_shape)
        self.batch_size = int(batch_size)
        self.device = torch.device(device)

        if model is None:
            from data_meta_map.models import get_model
            model = get_model(model_name, pretrained=pretrained)
        self.model = model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        children = list(self.model.children())
        if len(children) <= 1:
            self.extractor: nn.Module = self.model
        else:
            self.extractor = nn.Sequential(*children[:-1]).to(self.device).eval()

    @torch.inference_mode()
    def transform(self, X: torch.Tensor) -> torch.Tensor:
        X = X.float()
        c, h, w = self.image_shape
        if X.numel() % (c * h * w) != 0:
            raise ValueError(
                f"PretrainedEncoder cannot reshape inputs of dim "
                f"{X.shape[-1]} into image_shape={self.image_shape}"
            )
        X_img = X.view(-1, c, h, w)

        feats = []
        for start in range(0, X_img.shape[0], self.batch_size):
            chunk = X_img[start : start + self.batch_size].to(self.device)
            out = self.extractor(chunk)
            feats.append(out.view(out.size(0), -1).cpu())
        return torch.cat(feats, dim=0)


class _AE(nn.Module):
    """Tiny fully-connected autoencoder used by :class:`GenerativeEncoder`."""

    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z


class GenerativeEncoder(Encoder):
    """Per-dataset autoencoder trained on the input sample.

    The motivation comes from the MMD-as-generative-model perspective
    discussed in section 3.5.2 of MMD.pdf: rather than comparing raw
    distributions, we compare them in the latent space of a small
    generative model fit to each dataset. The implementation here is
    intentionally minimal -- a 1-hidden-layer MLP autoencoder -- so it
    has no extra dependencies and runs on CPU within a few seconds for
    moderate samples.

    Args:
        latent_dim: Dimensionality of the bottleneck used as features.
        hidden_dim: Width of the AE's hidden layer.
        epochs: Number of training epochs per dataset.
        batch_size: Mini-batch size.
        lr: Adam learning rate.
        device: Torch device.
        seed: Optional seed for reproducibility.
    """

    def __init__(
        self,
        latent_dim: int = 32,
        hidden_dim: int = 128,
        *,
        epochs: int = 5,
        batch_size: int = 128,
        lr: float = 1e-3,
        device: Union[str, torch.device] = "cpu",
        seed: Optional[int] = None,
    ):
        if latent_dim <= 0 or hidden_dim <= 0:
            raise ValueError("latent_dim and hidden_dim must be positive")
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.device = torch.device(device)
        self.seed = seed
        self._model: Optional[_AE] = None
        self._input_dim: Optional[int] = None

    def fit(self, X: torch.Tensor) -> "GenerativeEncoder":
        if self.seed is not None:
            torch.manual_seed(int(self.seed))
        X = X.float().to(self.device)
        self._input_dim = X.shape[1]
        self._model = _AE(self._input_dim, self.latent_dim, self.hidden_dim).to(self.device)
        opt = torch.optim.Adam(self._model.parameters(), lr=self.lr)

        n = X.shape[0]
        for _ in range(self.epochs):
            perm = torch.randperm(n, device=self.device)
            for start in range(0, n, self.batch_size):
                idx = perm[start : start + self.batch_size]
                batch = X[idx]
                opt.zero_grad()
                x_hat, _ = self._model(batch)
                loss = F.mse_loss(x_hat, batch)
                loss.backward()
                opt.step()
        self._model.eval()
        return self

    @torch.inference_mode()
    def transform(self, X: torch.Tensor) -> torch.Tensor:
        if self._model is None or self._input_dim is None:
            self.fit(X)
        if X.shape[1] != self._input_dim:
            raise ValueError(
                f"GenerativeEncoder was fit on dim {self._input_dim}, "
                f"got input of dim {X.shape[1]}"
            )
        X = X.float().to(self.device)
        feats = []
        for start in range(0, X.shape[0], self.batch_size):
            chunk = X[start : start + self.batch_size]
            _, z = self._model(chunk)
            feats.append(z.cpu())
        return torch.cat(feats, dim=0)
