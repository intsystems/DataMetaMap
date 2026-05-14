"""Top-level Maximum Mean Discrepancy embedder.

Wraps the kernels, estimators, RFF mean embedding and encoder backends
provided by :mod:`data_meta_map.mmd` into a single
:class:`MMDEmbedder` class implementing the
:class:`data_meta_map.base_embedder.BaseEmbedder` interface.

Two output modes are supported:

* ``mode="distance"`` -- compute the symmetric ``MMD^2`` matrix
  between every pair of input datasets and embed the result with
  classical metric MDS (mirrors how
  :class:`data_meta_map.wasserstein_embedder.WassersteinEmbedder`
  embeds class-wise distances).
* ``mode="rff"`` -- represent each dataset by its empirical kernel
  mean embedding ``mu_hat_P`` evaluated through Random Fourier
  Features (eq. 3.27 of MMD.pdf). Distances reduce to plain Euclidean
  norms in this representation.

Both modes share the same ``encoder=`` argument, which selects the
auxiliary representation passed to MMD: raw flattened features
(:class:`~data_meta_map.mmd.encoders.RawEncoder`), a frozen pretrained
network (:class:`~data_meta_map.mmd.encoders.PretrainedEncoder`), or a
per-dataset autoencoder
(:class:`~data_meta_map.mmd.encoders.GenerativeEncoder`).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from sklearn.manifold import MDS
from torch.utils.data import DataLoader, Dataset

from data_meta_map.base_embedder import BaseEmbedder
from data_meta_map.mmd.encoders import Encoder, RawEncoder
from data_meta_map.mmd.kernels import Kernel, get_kernel, median_heuristic
from data_meta_map.mmd.mmd import mmd2_biased, mmd2_linear, mmd2_unbiased
from data_meta_map.mmd.rff import RFFKernelMean
from data_meta_map.mmd.utils import dataset_to_tensor


_ESTIMATORS = {
    "biased": mmd2_biased,
    "unbiased": mmd2_unbiased,
    "linear": mmd2_linear,
}


def mmd(
    data_a: Union[Dataset, DataLoader, torch.Tensor],
    data_b: Union[Dataset, DataLoader, torch.Tensor],
    *,
    kernel: Union[str, Kernel] = "rbf",
    estimator: str = "unbiased",
    encoder: Optional[Encoder] = None,
    bandwidth: Union[str, float, None] = "median",
    max_samples: Optional[int] = None,
    device: Union[str, torch.device] = "cpu",
) -> torch.Tensor:
    """Convenience function: compute ``MMD^2`` between two datasets / tensors.

    Mirrors the role of ``task2vec()`` and ``dataset2vec()`` -- a
    one-liner that hides the embedder construction. For repeated calls
    on more than two datasets, prefer building a :class:`MMDEmbedder`
    once and reusing its caches.

    Args:
        data_a: First dataset, dataloader or pre-flattened tensor.
        data_b: Second dataset, dataloader or pre-flattened tensor.
        kernel: Kernel name or instance.
        estimator: ``"biased"``, ``"unbiased"`` or ``"linear"``.
        encoder: Optional auxiliary representation backend.
        bandwidth: ``"median"`` for the median heuristic on the joint
            sample, a positive float, or ``None`` to defer to the
            kernel default.
        max_samples: Optional subsampling cap per dataset.
        device: Computation device.

    Returns:
        Scalar tensor with the chosen MMD^2 estimate.
    """
    embedder = MMDEmbedder(
        mode="distance",
        kernel=kernel,
        estimator=estimator,
        encoder=encoder,
        bandwidth=bandwidth,
        max_samples=max_samples,
        device=device,
    )
    Xa = embedder._materialize(data_a, dataset_id=None)
    Xb = embedder._materialize(data_b, dataset_id=None)
    return embedder._mmd2(Xa, Xb)


class MMDEmbedder(BaseEmbedder):
    """Maximum Mean Discrepancy dataset embedder.

    Args:
        mode: Output strategy.
            * ``"distance"`` -- pairwise MMD^2 matrix + MDS into
              :math:`\\mathbb{R}^{\\text{emb\\_dim}}`.
            * ``"rff"`` -- Random Fourier Features mean embedding
              :math:`\\hat\\mu_P \\in \\mathbb{R}^{n\\_rff}`.
        kernel: Kernel name (``"rbf"`` / ``"linear"`` / ``"poly"`` /
            ``"imq"``) or a :class:`~data_meta_map.mmd.kernels.Kernel`
            instance. Only ``"rbf"`` is supported in ``mode="rff"``.
        bandwidth: ``"median"`` to use the median heuristic on the
            joint sample at fit time, a positive float, or ``None``.
        encoder: Optional auxiliary representation backend
            (:class:`RawEncoder` is used when ``None``).
        emb_dim: Dimensionality of the MDS embedding (only used in
            ``mode="distance"``).
        n_rff: Dimensionality of the random feature map (only used in
            ``mode="rff"``).
        estimator: Empirical estimator: ``"biased"``, ``"unbiased"``
            or ``"linear"``.
        max_samples: Optional subsampling cap per dataset.
        batch_size: Batch size used when materializing datasets.
        device: Computation device.
        seed: Optional random seed for the RFF projection / linear
            estimator.
    """

    _VALID_MODES = ("distance", "rff")

    def __init__(
        self,
        mode: str = "distance",
        *,
        kernel: Union[str, Kernel] = "rbf",
        bandwidth: Union[str, float, None] = "median",
        encoder: Optional[Encoder] = None,
        emb_dim: int = 2,
        n_rff: int = 512,
        estimator: str = "unbiased",
        max_samples: Optional[int] = 2000,
        batch_size: int = 64,
        device: Union[str, torch.device] = "cpu",
        seed: Optional[int] = None,
    ):
        super().__init__()
        if mode not in self._VALID_MODES:
            raise ValueError(
                f"mode must be one of {self._VALID_MODES}, got '{mode}'"
            )
        if estimator not in _ESTIMATORS:
            raise ValueError(
                f"estimator must be one of {list(_ESTIMATORS)}, got '{estimator}'"
            )

        self.mode = mode
        self.kernel_spec = kernel
        self.bandwidth = bandwidth
        self.encoder: Encoder = encoder if encoder is not None else RawEncoder()
        self.emb_dim = int(emb_dim)
        self.n_rff = int(n_rff)
        self.estimator = estimator
        self.max_samples = max_samples
        self.batch_size = int(batch_size)
        self.device = torch.device(device) if isinstance(device, str) else device
        self.seed = seed

        self._kernel: Optional[Kernel] = None
        self._rff: Optional[RFFKernelMean] = None
        self._feat_cache: Dict[int, torch.Tensor] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _materialize(
        self,
        data: Union[Dataset, DataLoader, torch.Tensor],
        dataset_id: Optional[int],
    ) -> torch.Tensor:
        """Return cached encoded features for ``data``."""
        if dataset_id is not None and dataset_id in self._feat_cache:
            return self._feat_cache[dataset_id]

        if isinstance(data, torch.Tensor):
            X = data.float().view(data.shape[0], -1).to(self.device)
        else:
            X, _ = dataset_to_tensor(
                data,
                batch_size=self.batch_size,
                max_samples=self.max_samples,
                device=self.device,
                return_labels=False,
            )

        feats = self.encoder.fit(X).transform(X)
        feats = feats.float().to(self.device)

        if dataset_id is not None:
            self._feat_cache[dataset_id] = feats
        return feats

    def _resolve_kernel(self, ref: torch.Tensor) -> Kernel:
        if self._kernel is not None:
            return self._kernel
        kwargs = {}
        if isinstance(self.kernel_spec, str) and self.kernel_spec.lower() in (
            "rbf",
            "gaussian",
        ):
            if isinstance(self.bandwidth, (int, float)):
                kwargs["sigma"] = float(self.bandwidth)
            elif self.bandwidth == "median":
                kwargs["sigma"] = float(median_heuristic(ref))
        self._kernel = get_kernel(self.kernel_spec, **kwargs)
        return self._kernel

    def _mmd2(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        joint = torch.cat([X, Y], dim=0)
        kernel = self._resolve_kernel(joint)
        fn = _ESTIMATORS[self.estimator]
        if self.estimator == "linear":
            gen = (
                torch.Generator(device="cpu").manual_seed(int(self.seed))
                if self.seed is not None
                else None
            )
            return fn(X, Y, kernel=kernel, generator=gen)
        return fn(X, Y, kernel=kernel)

    def _ensure_rff(self, ref: torch.Tensor) -> RFFKernelMean:
        if self._rff is not None:
            return self._rff
        if not (
            isinstance(self.kernel_spec, str)
            and self.kernel_spec.lower() in ("rbf", "gaussian")
        ):
            raise ValueError(
                "mode='rff' currently supports only the RBF kernel; "
                f"got kernel='{self.kernel_spec}'"
            )
        if isinstance(self.bandwidth, (int, float)):
            sigma = float(self.bandwidth)
        elif self.bandwidth == "median":
            sigma = float(median_heuristic(ref))
        else:
            sigma = None
        self._rff = RFFKernelMean(
            d=ref.shape[1],
            n_features=self.n_rff,
            sigma=sigma,
            seed=self.seed,
            device=self.device,
        )
        return self._rff

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def preprocess_dataset(
        self,
        data: Union[Dataset, DataLoader, torch.Tensor],
        dataset_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Materialize a dataset into encoded feature tensor of shape ``[N, D]``."""
        return self._materialize(data, dataset_id=dataset_id)

    def compute_pairwise_distances(
        self,
        datasets: List[Union[Dataset, DataLoader, torch.Tensor]],
        symmetric: bool = True,
    ) -> torch.Tensor:
        """Symmetric ``MMD^2`` matrix between every pair of datasets.

        Args:
            datasets: List of ``K`` datasets / dataloaders / tensors.
            symmetric: Compute only the upper triangle and mirror it
                (``True`` by default; off-diagonal estimates are mathe-
                matically symmetric anyway).

        Returns:
            Tensor of shape ``[K, K]``.
        """
        feats = [self._materialize(d, dataset_id=i) for i, d in enumerate(datasets)]
        K = len(feats)
        D = torch.zeros((K, K), device=self.device)
        for i in range(K):
            for j in range(i if symmetric else 0, K):
                if i == j:
                    D[i, j] = 0.0
                    continue
                d = self._mmd2(feats[i], feats[j])
                d = torch.clamp(d, min=0.0)
                D[i, j] = d
                if symmetric:
                    D[j, i] = d
        return D

    def embed_distance_matrix(
        self,
        distance_matrix: torch.Tensor,
        emb_dim: Optional[int] = None,
    ) -> torch.Tensor:
        """Embed a ``[K, K]`` MMD^2 matrix into ``R^emb_dim`` via metric MDS."""
        target_dim = emb_dim if emb_dim is not None else self.emb_dim
        D_np = distance_matrix.detach().cpu().numpy()
        np.fill_diagonal(D_np, 0.0)
        D_np = (D_np + D_np.T) / 2.0
        # MDS expects distances, not squared distances
        D_np = np.sqrt(np.clip(D_np, 0.0, None))
        if D_np.shape[0] < 2:
            return torch.zeros(D_np.shape[0], target_dim, device=self.device)
        mds = MDS(
            n_components=target_dim,
            dissimilarity="precomputed",
            n_init=4,
            max_iter=1000,
            random_state=42,
            normalized_stress="auto",
        )
        embs = mds.fit_transform(D_np)
        return torch.from_numpy(embs).to(self.device).float()

    def embed(
        self,
        datasets: List[Union[Dataset, DataLoader, torch.Tensor]],
        **kwargs,
    ) -> torch.Tensor:
        """Compute per-dataset embeddings.

        Args:
            datasets: List of ``K`` datasets / dataloaders / tensors.
            **kwargs: ``emb_dim=`` overrides the configured MDS
                dimensionality (``mode="distance"`` only).

        Returns:
            Tensor of shape ``[K, emb_dim]`` in ``mode="distance"`` or
            ``[K, n_rff]`` in ``mode="rff"``.
        """
        if self.mode == "distance":
            D = self.compute_pairwise_distances(datasets, symmetric=True)
            return self.embed_distance_matrix(D, emb_dim=kwargs.get("emb_dim"))

        feats = [self._materialize(d, dataset_id=i) for i, d in enumerate(datasets)]
        joint = torch.cat(feats, dim=0)
        rff = self._ensure_rff(joint)
        return torch.stack([rff.embed(X) for X in feats], dim=0).to(self.device)

    def clear_cache(self) -> None:
        """Free cached features, kernels and RFF projections."""
        self._feat_cache.clear()
        self._kernel = None
        self._rff = None
