"""Positive-definite kernels for MMD computation.

Each kernel returns either a scalar value ``k(x, y)`` for two single
vectors or a Gram matrix ``K[i, j] = k(x_i, y_j)`` when called on two
batches. The classes are deliberately lightweight so they can be passed
around as configuration objects.

The bandwidth selection helper :func:`median_heuristic` follows
Gretton et al. 2005b (see also p. 57 of MMD.pdf):

    sigma^2 = median { ||x_i - x_j||^2 : i, j = 1, ..., n }.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Union

import torch


def _pairwise_sq_dists(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Compute pairwise squared Euclidean distances ``||x_i - y_j||^2``.

    Args:
        X: Tensor of shape ``[n, d]``.
        Y: Tensor of shape ``[m, d]``.

    Returns:
        Tensor of shape ``[n, m]`` with entries ``||x_i - y_j||^2``,
        clamped at ``0`` for numerical safety.
    """
    if X.dim() != 2 or Y.dim() != 2:
        raise ValueError(
            f"Expected 2-D inputs, got X.dim()={X.dim()}, Y.dim()={Y.dim()}"
        )
    if X.shape[1] != Y.shape[1]:
        raise ValueError(
            f"Feature dimensions must match: {X.shape[1]} vs {Y.shape[1]}"
        )
    XX = (X * X).sum(dim=1, keepdim=True)            # [n, 1]
    YY = (Y * Y).sum(dim=1, keepdim=True).T          # [1, m]
    XY = X @ Y.T                                     # [n, m]
    sq = XX + YY - 2.0 * XY
    return sq.clamp(min=0.0)


class Kernel(ABC):
    """Abstract positive-definite kernel."""

    @abstractmethod
    def __call__(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Return the Gram matrix ``K[i, j] = k(x_i, y_j)``."""

    def diag(self, X: torch.Tensor) -> torch.Tensor:
        """Return the diagonal ``k(x_i, x_i)`` for each row of ``X``.

        Default implementation evaluates the full Gram matrix and
        extracts its diagonal; subclasses with cheap closed forms
        (e.g. RBF) may override this for efficiency.
        """
        return torch.diagonal(self(X, X))


class RBFKernel(Kernel):
    """Gaussian / RBF kernel ``k(x, y) = exp(-||x - y||^2 / (2 sigma^2))``.

    Args:
        sigma: Bandwidth. If ``None``, the median heuristic is applied
            on the first batch passed to ``__call__``.
    """

    def __init__(self, sigma: Optional[float] = None):
        self.sigma = sigma

    def __call__(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        sq = _pairwise_sq_dists(X, Y)
        sigma = self.sigma
        if sigma is None:
            sigma = float(median_heuristic(X))
            self.sigma = sigma
        sigma2 = max(float(sigma) ** 2, 1e-12)
        return torch.exp(-sq / (2.0 * sigma2))

    def diag(self, X: torch.Tensor) -> torch.Tensor:
        return torch.ones(X.shape[0], device=X.device, dtype=X.dtype)


class LinearKernel(Kernel):
    """Linear kernel ``k(x, y) = <x, y>``."""

    def __call__(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        return X @ Y.T


class PolynomialKernel(Kernel):
    """Polynomial kernel ``k(x, y) = (gamma <x, y> + coef0)^degree``.

    Args:
        degree: Polynomial degree (must be a positive integer).
        gamma: Scale on the inner product. ``None`` defaults to
            ``1 / d`` where ``d`` is the feature dimension.
        coef0: Additive constant.
    """

    def __init__(
        self,
        degree: int = 3,
        gamma: Optional[float] = None,
        coef0: float = 1.0,
    ):
        if degree < 1:
            raise ValueError("degree must be >= 1")
        self.degree = int(degree)
        self.gamma = gamma
        self.coef0 = float(coef0)

    def __call__(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        gamma = self.gamma
        if gamma is None:
            gamma = 1.0 / X.shape[1]
        return (gamma * (X @ Y.T) + self.coef0) ** self.degree


class IMQKernel(Kernel):
    """Inverse multiquadric kernel ``k(x, y) = (c^2 + ||x - y||^2)^(-beta)``.

    Useful as a heavier-tailed alternative to the RBF kernel; commonly
    used in Stein discrepancies and MMD-GAN literature.

    Args:
        c: Bias term inside the parentheses.
        beta: Positive exponent (default 0.5 yields the classical IMQ).
    """

    def __init__(self, c: float = 1.0, beta: float = 0.5):
        if beta <= 0:
            raise ValueError("beta must be > 0")
        self.c = float(c)
        self.beta = float(beta)

    def __call__(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        sq = _pairwise_sq_dists(X, Y)
        return (self.c ** 2 + sq) ** (-self.beta)


def median_heuristic(X: torch.Tensor, Y: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Median-heuristic bandwidth ``sqrt(median ||x_i - x_j||^2)``.

    Args:
        X: Tensor of shape ``[n, d]``.
        Y: Optional second tensor; if provided, the median is computed
            over the joint sample ``[X; Y]``.

    Returns:
        Scalar tensor with the bandwidth ``sigma > 0``. A small floor
        is enforced to keep RBF-style kernels numerically well-defined
        when the sample collapses to a point.
    """
    if Y is not None:
        Z = torch.cat([X, Y], dim=0)
    else:
        Z = X
    sq = _pairwise_sq_dists(Z, Z)
    n = sq.shape[0]
    if n < 2:
        return torch.tensor(1.0, device=X.device, dtype=X.dtype)
    iu = torch.triu_indices(n, n, offset=1)
    upper = sq[iu[0], iu[1]]
    upper = upper[upper > 0]
    if upper.numel() == 0:
        return torch.tensor(1.0, device=X.device, dtype=X.dtype)
    med = torch.median(upper)
    sigma = torch.sqrt(med).clamp(min=1e-6)
    return sigma


_KERNELS = {
    "rbf": RBFKernel,
    "gaussian": RBFKernel,
    "linear": LinearKernel,
    "poly": PolynomialKernel,
    "polynomial": PolynomialKernel,
    "imq": IMQKernel,
}


def get_kernel(name: Union[str, Kernel], **kwargs) -> Kernel:
    """Resolve a kernel by name or pass through a :class:`Kernel` instance.

    Args:
        name: Either a string identifier (``"rbf"``, ``"linear"``,
            ``"poly"``, ``"imq"``) or an already-instantiated kernel.
        **kwargs: Forwarded to the kernel constructor.

    Returns:
        A :class:`Kernel` instance.
    """
    if isinstance(name, Kernel):
        return name
    key = name.lower()
    if key not in _KERNELS:
        raise ValueError(
            f"Unknown kernel '{name}'. Available: {sorted(_KERNELS)}"
        )
    return _KERNELS[key](**kwargs)
