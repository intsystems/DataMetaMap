"""Maximum Mean Discrepancy (MMD) subpackage.

Implements the kernel-based two-sample MMD estimators, Random Fourier
Feature kernel mean embeddings, and pluggable representation backends
(raw / pretrained encoder / generative AE latent) used by
:class:`data_meta_map.mmd_embedder.MMDEmbedder`.

References
----------
Gretton, A. et al. *A Kernel Two-Sample Test* (JMLR 2012). The
estimators here follow eq. (3.30) and eq. (3.32) of the
"Hilbert Space Embedding of Marginal Distributions" review
(Muandet et al., 2017).
"""

from .kernels import (
    Kernel,
    RBFKernel,
    LinearKernel,
    PolynomialKernel,
    IMQKernel,
    get_kernel,
    median_heuristic,
)
from .mmd import mmd2_biased, mmd2_unbiased, mmd2_linear, gram_matrix
from .rff import RFFKernelMean, rff_features
from .encoders import Encoder, RawEncoder, PretrainedEncoder, GenerativeEncoder

__all__ = [
    "Kernel",
    "RBFKernel",
    "LinearKernel",
    "PolynomialKernel",
    "IMQKernel",
    "get_kernel",
    "median_heuristic",
    "mmd2_biased",
    "mmd2_unbiased",
    "mmd2_linear",
    "gram_matrix",
    "RFFKernelMean",
    "rff_features",
    "Encoder",
    "RawEncoder",
    "PretrainedEncoder",
    "GenerativeEncoder",
]
