***
MMD
***

Maximum Mean Discrepancy (MMD) is a kernel-based, non-parametric distance
between probability distributions. The :mod:`data_meta_map.mmd` subpackage
provides the kernel zoo, biased / unbiased / linear-time MMD\ :sup:`2`
estimators and a Random Fourier Features kernel mean embedding, while
:class:`data_meta_map.mmd_embedder.MMDEmbedder` ties everything together
behind the :class:`~data_meta_map.base_embedder.BaseEmbedder` interface
and exposes three pluggable representation backends -- raw features, a
frozen pretrained encoder, and a small per-dataset autoencoder.

MMDEmbedder
===========

.. automodule:: data_meta_map.mmd_embedder
    :members:
    :undoc-members:
    :show-inheritance:

Kernels
=======

.. automodule:: data_meta_map.mmd.kernels
    :members:
    :undoc-members:
    :show-inheritance:

Empirical estimators
====================

.. automodule:: data_meta_map.mmd.mmd
    :members:
    :undoc-members:
    :show-inheritance:

Random Fourier Features
=======================

.. automodule:: data_meta_map.mmd.rff
    :members:
    :undoc-members:
    :show-inheritance:

Representation backends
=======================

.. automodule:: data_meta_map.mmd.encoders
    :members:
    :undoc-members:
    :show-inheritance:

Utilities
=========

.. automodule:: data_meta_map.mmd.utils
    :members:
    :undoc-members:
    :show-inheritance:
