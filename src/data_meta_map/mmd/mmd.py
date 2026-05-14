"""Empirical MMD^2 estimators between two samples.

The expressions below mirror the formulas in Section 3.5 of MMD.pdf
(Muandet et al., 2017):

* eq. (3.30) -- biased (V-statistic) estimator
  ``MMD^2_b = E[k(X,X')] - 2 E[k(X,Y)] + E[k(Y,Y')]``,
* eq. (3.32) -- unbiased (U-statistic) estimator
  (drops self-pair diagonals from the X- and Y-only terms),
* a linear-time variant from Gretton et al. (2012a, sec. 6 / sec. 3.5.1
  of MMD.pdf), based on subsampling pairs without replacement.
"""

from __future__ import annotations

from typing import Optional, Union

import torch

from .kernels import Kernel, get_kernel


def gram_matrix(
    X: torch.Tensor,
    Y: torch.Tensor,
    kernel: Union[str, Kernel] = "rbf",
    **kernel_kwargs,
) -> torch.Tensor:
    """Compute the Gram matrix ``K[i, j] = k(x_i, y_j)``.

    Args:
        X: Tensor of shape ``[n, d]``.
        Y: Tensor of shape ``[m, d]``.
        kernel: Either a kernel name (resolved via :func:`get_kernel`)
            or an already-instantiated :class:`Kernel`.
        **kernel_kwargs: Forwarded to :func:`get_kernel` when ``kernel``
            is a string.

    Returns:
        Tensor of shape ``[n, m]``.
    """
    k = get_kernel(kernel, **kernel_kwargs)
    return k(X, Y)


def _resolve_kernel(kernel: Union[str, Kernel], **kernel_kwargs) -> Kernel:
    return get_kernel(kernel, **kernel_kwargs)


def mmd2_biased(
    X: torch.Tensor,
    Y: torch.Tensor,
    kernel: Union[str, Kernel] = "rbf",
    **kernel_kwargs,
) -> torch.Tensor:
    """Biased empirical estimate of MMD^2 (eq. 3.30 of MMD.pdf).

    Includes self-pairs ``k(x_i, x_i)``, which biases the estimator
    upward but keeps it non-negative for any positive-definite kernel.

    Args:
        X: Tensor of shape ``[n, d]`` drawn from distribution P.
        Y: Tensor of shape ``[m, d]`` drawn from distribution Q.
        kernel: Kernel name or :class:`Kernel` instance.
        **kernel_kwargs: Forwarded to :func:`get_kernel`.

    Returns:
        Scalar tensor with the biased MMD^2 estimate.
    """
    k = _resolve_kernel(kernel, **kernel_kwargs)
    Kxx = k(X, X)
    Kyy = k(Y, Y)
    Kxy = k(X, Y)
    return Kxx.mean() + Kyy.mean() - 2.0 * Kxy.mean()


def mmd2_unbiased(
    X: torch.Tensor,
    Y: torch.Tensor,
    kernel: Union[str, Kernel] = "rbf",
    **kernel_kwargs,
) -> torch.Tensor:
    """Unbiased empirical estimate of MMD^2 (eq. 3.32 of MMD.pdf).

    Drops the diagonal ``k(x_i, x_i)`` and ``k(y_i, y_i)`` terms.
    Can be slightly negative for finite samples drawn from the same
    distribution.

    Args:
        X: Tensor of shape ``[n, d]``, ``n >= 2``.
        Y: Tensor of shape ``[m, d]``, ``m >= 2``.
        kernel: Kernel name or :class:`Kernel` instance.
        **kernel_kwargs: Forwarded to :func:`get_kernel`.

    Returns:
        Scalar tensor with the unbiased MMD^2 estimate.
    """
    n = X.shape[0]
    m = Y.shape[0]
    if n < 2 or m < 2:
        raise ValueError("Unbiased MMD^2 requires at least 2 samples per group")
    k = _resolve_kernel(kernel, **kernel_kwargs)
    Kxx = k(X, X)
    Kyy = k(Y, Y)
    Kxy = k(X, Y)

    sum_xx = Kxx.sum() - torch.diagonal(Kxx).sum()
    sum_yy = Kyy.sum() - torch.diagonal(Kyy).sum()
    sum_xy = Kxy.sum()

    term_xx = sum_xx / (n * (n - 1))
    term_yy = sum_yy / (m * (m - 1))
    term_xy = 2.0 * sum_xy / (n * m)
    return term_xx + term_yy - term_xy


def mmd2_linear(
    X: torch.Tensor,
    Y: torch.Tensor,
    kernel: Union[str, Kernel] = "rbf",
    generator: Optional[torch.Generator] = None,
    **kernel_kwargs,
) -> torch.Tensor:
    """Linear-time MMD^2 estimator (Gretton et al. 2012a).

    Both samples are paired into floor(min(n, m) / 2) disjoint quadruples
    ``(x_{2i-1}, x_{2i}, y_{2i-1}, y_{2i})`` and the per-quadruple
    statistic ``h(v_i, v_j)`` from sec. 3.5.1 of MMD.pdf is averaged.
    The result is unbiased but has higher variance than the quadratic-
    time version.

    Args:
        X: Tensor of shape ``[n, d]``, ``n >= 2``.
        Y: Tensor of shape ``[m, d]``, ``m >= 2``.
        kernel: Kernel name or :class:`Kernel` instance.
        generator: Optional torch generator for the pair shuffle.
        **kernel_kwargs: Forwarded to :func:`get_kernel`.

    Returns:
        Scalar tensor with the linear-time MMD^2 estimate.
    """
    n = X.shape[0]
    m = Y.shape[0]
    n_pairs = min(n, m) // 2
    if n_pairs < 1:
        raise ValueError(
            "Linear-time MMD^2 requires at least 2 samples per group"
        )

    perm_x = torch.randperm(n, generator=generator, device=X.device)[: 2 * n_pairs]
    perm_y = torch.randperm(m, generator=generator, device=Y.device)[: 2 * n_pairs]
    x1 = X[perm_x[0::2]]
    x2 = X[perm_x[1::2]]
    y1 = Y[perm_y[0::2]]
    y2 = Y[perm_y[1::2]]

    k = _resolve_kernel(kernel, **kernel_kwargs)
    h = (
        torch.diagonal(k(x1, x2))
        + torch.diagonal(k(y1, y2))
        - torch.diagonal(k(x1, y2))
        - torch.diagonal(k(x2, y1))
    )
    return h.mean()
