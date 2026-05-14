"""Random Fourier Features (RFF) approximation of RBF mean embeddings.

Implements eq. (3.27) of MMD.pdf:

    z(x) = sqrt(2 / m) * cos(W x + b),    W ~ N(0, sigma^{-2} I),  b ~ U[0, 2*pi),

so that ``<z(x), z(y)> ~= k(x, y)`` for a Gaussian RBF kernel with
bandwidth ``sigma``. The empirical kernel mean of a sample
``X = {x_1, ..., x_n}`` is then

    mu_hat_P = (1 / n) * sum_i z(x_i)  in R^m.

Storing ``mu_hat_P`` instead of ``X`` itself yields a compact,
fixed-size dataset embedding; the squared MMD between two samples
reduces to ``||mu_hat_P - mu_hat_Q||^2`` (a tight approximation when
``m`` is large).
"""

from __future__ import annotations

from typing import Optional

import torch

from .kernels import median_heuristic


def rff_features(
    X: torch.Tensor,
    n_features: int,
    sigma: Optional[float] = None,
    *,
    seed: Optional[int] = None,
    W: Optional[torch.Tensor] = None,
    b: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply a single random Fourier feature map to ``X``.

    Args:
        X: Tensor of shape ``[n, d]``.
        n_features: Output feature dimensionality ``m``.
        sigma: Bandwidth of the underlying RBF kernel. If ``None``,
            the median heuristic is used on ``X``.
        seed: Optional seed for the random projection. Ignored when
            ``W`` and ``b`` are provided.
        W: Optional pre-sampled projection matrix of shape ``[m, d]``.
        b: Optional pre-sampled bias vector of shape ``[m]``.

    Returns:
        Tensor of shape ``[n, m]``.
    """
    n, d = X.shape
    if sigma is None:
        sigma = float(median_heuristic(X))
    sigma = max(float(sigma), 1e-12)

    if W is None or b is None:
        gen = None
        if seed is not None:
            gen = torch.Generator(device="cpu").manual_seed(int(seed))
        W = torch.randn(n_features, d, generator=gen) / sigma
        b = 2.0 * torch.pi * torch.rand(n_features, generator=gen)
        W = W.to(device=X.device, dtype=X.dtype)
        b = b.to(device=X.device, dtype=X.dtype)
    else:
        W = W.to(device=X.device, dtype=X.dtype)
        b = b.to(device=X.device, dtype=X.dtype)

    proj = X @ W.T + b                                  # [n, m]
    return torch.sqrt(torch.tensor(2.0 / n_features, device=X.device, dtype=X.dtype)) * torch.cos(proj)


class RFFKernelMean:
    """Random Fourier Features kernel mean embedding (eq. 3.27 of MMD.pdf).

    The same projection ``(W, b)`` must be reused across datasets so
    that mean embeddings live in a common space. Construct the object
    once and call :meth:`embed` on each sample.

    Args:
        d: Input feature dimensionality.
        n_features: Output dimensionality of the random feature map.
        sigma: Bandwidth of the underlying RBF kernel. May also be set
            later via :meth:`fit_bandwidth`.
        seed: Optional random seed for reproducibility.
        device: Torch device used to materialize ``W`` and ``b``.
        dtype: Tensor dtype.
    """

    def __init__(
        self,
        d: int,
        n_features: int = 512,
        sigma: Optional[float] = None,
        *,
        seed: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        if d <= 0 or n_features <= 0:
            raise ValueError("d and n_features must be positive integers")
        self.d = int(d)
        self.n_features = int(n_features)
        self.sigma = sigma
        self.dtype = dtype
        self.device = device if device is not None else torch.device("cpu")

        gen = None
        if seed is not None:
            gen = torch.Generator(device="cpu").manual_seed(int(seed))
        # We store unscaled W to allow late binding of sigma.
        self._W_unit = torch.randn(self.n_features, self.d, generator=gen).to(
            device=self.device, dtype=self.dtype
        )
        self.b = (2.0 * torch.pi * torch.rand(self.n_features, generator=gen)).to(
            device=self.device, dtype=self.dtype
        )

    @property
    def W(self) -> torch.Tensor:
        sigma = self.sigma if self.sigma is not None else 1.0
        return self._W_unit / float(sigma)

    def fit_bandwidth(self, X: torch.Tensor) -> "RFFKernelMean":
        """Set ``sigma`` via the median heuristic on a reference sample.

        Args:
            X: Tensor of shape ``[n, d]``.

        Returns:
            ``self`` for chaining.
        """
        self.sigma = float(median_heuristic(X))
        return self

    def features(self, X: torch.Tensor) -> torch.Tensor:
        """Compute per-sample RFF features ``z(x_i)`` of shape ``[n, m]``."""
        if self.sigma is None:
            self.fit_bandwidth(X)
        proj = X.to(self.dtype) @ self.W.T + self.b
        scale = torch.sqrt(torch.tensor(2.0 / self.n_features, device=X.device, dtype=self.dtype))
        return scale * torch.cos(proj)

    def embed(self, X: torch.Tensor) -> torch.Tensor:
        """Empirical kernel mean ``mu_hat_P = mean_i z(x_i)`` of shape ``[m]``."""
        return self.features(X).mean(dim=0)

    def embed_many(self, samples) -> torch.Tensor:
        """Stack mean embeddings of multiple samples into a ``[K, m]`` tensor."""
        return torch.stack([self.embed(s) for s in samples], dim=0)
