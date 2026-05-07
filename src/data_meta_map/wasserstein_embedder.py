from abc import ABC
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from sklearn.manifold import MDS
import ot  # POT: Python Optimal Transport
from tqdm.autonotebook import tqdm

from data_meta_map.base_embedder import BaseEmbedder


def sqrtm_newton_schulz(A: torch.Tensor, num_iters: int = 20) -> torch.Tensor:
    """
    Matrix square root via Newton-Schulz iteration.
    Adapted from OTDD (https://github.com/microsoft/otdd)

    Args:
        A: Square positive semi-definite matrix [d, d]
        num_iters: Number of Newton-Schulz iterations

    Returns:
        sqrtA: Matrix square root [d, d]
    """
    dim = A.shape[0]

    # Frobenius norm for stable normalization
    normA = torch.norm(A, p='fro')

    # Normalize to ensure convergence
    Y = A / normA
    I = torch.eye(dim, device=A.device, dtype=A.dtype)
    Z = torch.eye(dim, device=A.device, dtype=A.dtype)

    # Newton-Schulz iterations
    for _ in range(num_iters):
        T = 0.5 * (3.0 * I - Z @ Y)
        Y = Y @ T
        Z = T @ Z

    # Rescale back
    sqrtA = Y * torch.sqrt(normA)
    return sqrtA


def compute_bures_term(
    cov1: torch.Tensor,
    cov2: torch.Tensor,
    sqrt_cov1: Optional[torch.Tensor] = None,
    diagonal_cov: bool = False,
    num_iters: int = 20
) -> torch.Tensor:
    """
    Compute the covariance term of Bures-Wasserstein distance:
        Tr(Σ₁ + Σ₂ - 2(Σ₁^{1/2} Σ₂ Σ₁^{1/2})^{1/2})

    Args:
        cov1, cov2: Covariance matrices [d, d] or diagonals [d] if diagonal_cov=True
        sqrt_cov1: Precomputed sqrt(cov1) for efficiency
        diagonal_cov: If True, treat covariances as diagonal
        num_iters: Newton-Schulz iterations for matrix sqrt

    Returns:
        bures_term: Scalar tensor
    """
    if diagonal_cov:
        # Diagonal case: Tr(Σ₁ + Σ₂ - 2√(Σ₁Σ₂))
        return torch.sum(cov1 + cov2 - 2 * torch.sqrt(cov1 * cov2 + 1e-12))
    else:
        # Full matrix case
        if sqrt_cov1 is None:
            sqrt_cov1 = sqrtm_newton_schulz(cov1, num_iters=num_iters)

        # Compute (Σ₁^{1/2} Σ₂ Σ₁^{1/2})^{1/2}
        middle = sqrt_cov1 @ cov2 @ sqrt_cov1
        sqrt_middle = sqrtm_newton_schulz(middle, num_iters=num_iters)

        # Trace term
        return torch.trace(cov1 + cov2 - 2 * sqrt_middle)


class WassersteinEmbedder(BaseEmbedder):
    """
    Dataset embedder based on Wasserstein distance (Optimal Transport).

    Supports two distance computation modes:
        1. Gaussian approximation → Bures-Wasserstein distance (fast, O(d³))
        2. Exact OT via EMD (slow, O(n³ log n), but distribution-agnostic)

    Key Features:
        - Support for varying number of classes across datasets
        - Automatic caching of class statistics
        - Integration with POT library (Python Optimal Transport)
        - Optional diagonal covariance for speedup on high-dimensional data
        - Industrial-grade matrix sqrt implementation (OTDD/Microsoft)
    """

    def __init__(
        self,
        emb_dim: int = 2,
        device: Union[str, torch.device] = "cpu",
        max_samples: Optional[int] = None,
        batch_size: int = 64,
        gaussian_assumption: bool = True,
        diagonal_cov: bool = False,
        commute: bool = False,
        # 'ns' = Newton-Schulz (default), 'eig' = eigenvalue decomposition
        sqrt_method: str = "ns",
        sqrt_niters: int = 20,
        **kwargs
    ):
        """
        Initialize Wasserstein-based embedder.

        Args:
            emb_dim: Target dimensionality of label embeddings.
            device: Computation device ('cpu', 'cuda', or torch.device).
            max_samples: Maximum number of samples to process from dataset.
                         If None, all samples are used.
            batch_size: Batch size for DataLoader during data loading.
            gaussian_assumption: If True, use Gaussian approximation and Bures distance.
                                 If False, compute exact distance via EMD.
            diagonal_cov: Use only diagonal of covariance matrix (speedup).
            commute: Flag for commuting approximation of Bures distance (experimental).
            sqrt_method: Method for matrix square root computation ('ns', 'eig').
            sqrt_niters: Number of iterations for Newton-Schulz method.
        """
        super().__init__()
        self.emb_dim = emb_dim
        self.device = torch.device(device) if isinstance(device, str) else device
        self.max_samples = max_samples
        self.batch_size = batch_size
        self.gaussian_assumption = gaussian_assumption
        self.diagonal_cov = diagonal_cov
        self.commute = commute
        self.sqrt_method = sqrt_method
        self.sqrt_niters = sqrt_niters

        # Cache for class statistics: {dataset_id: (means, covs, class_offsets)}
        self._stats_cache: Dict[int,
                                Tuple[torch.Tensor, torch.Tensor, List[int]]] = {}
        # Cache for preprocessed data: {dataset_id: (X, Y)}
        self._data_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

    def preprocess_dataset(
        self,
        data: Union[Dataset, DataLoader],
        dataset_id: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Transform dataset/dataloader into feature-label tensor pair.

        Args:
             Object supporting either:
                  - Dataset interface: must return (features, label) in __getitem__
                  - DataLoader interface: must yield batches (X_batch, y_batch)
            dataset_id: Optional ID for caching results.

        Returns:
            X: Feature tensor of shape [num_samples, feature_dim], dtype=torch.float32
            Y: Label tensor of shape [num_samples], dtype=torch.long

        Notes:
            - Automatically creates DataLoader when Dataset is provided
            - Applies subsampling if self.max_samples is set
            - Flattens features to [N, D] format via .view(..., -1)
            - Results are cached when dataset_id is provided
        """
        # Check cache
        if dataset_id is not None and dataset_id in self._data_cache:
            return self._data_cache[dataset_id]

        # Create loader if necessary
        if isinstance(data, Dataset):
            if self.max_samples and len(data) > self.max_samples:
                idxs = np.sort(np.random.choice(
                    len(data), self.max_samples, replace=False))
                sampler = SubsetRandomSampler(idxs)
                loader = DataLoader(data, sampler=sampler,
                                    batch_size=self.batch_size)
            else:
                loader = DataLoader(
                    data, batch_size=self.batch_size, shuffle=False)
        elif isinstance(data, DataLoader):
            loader = data
        else:
            raise TypeError(
                f"Expected Dataset or DataLoader, got {type(data).__name__}"
            )

        # Aggregate data
        X_list: List[torch.Tensor] = []
        Y_list: List[torch.Tensor] = []

        for batch in tqdm(loader, desc="Preprocessing dataset", leave=False):
            x_batch = batch[0]  # [B, ...]
            y_batch = batch[1]  # [B]

            # Flatten to [B, D]
            x_flat = x_batch.view(x_batch.size(0), -1).float()
            y_flat = y_batch.long().view(-1)

            X_list.append(x_flat.to(self.device))
            Y_list.append(y_flat.to(self.device))

        X = torch.cat(X_list, dim=0)
        Y = torch.cat(Y_list, dim=0)

        # Cache results
        if dataset_id is not None:
            self._data_cache[dataset_id] = (X, Y)

        return X, Y

    def _compute_gaussian_stats(
        self,
        X: torch.Tensor,
        Y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """
        Compute Gaussian statistics (mean, covariance) for each class.

        Args:
            X: Feature tensor [num_samples, feature_dim]
            Y: Label tensor [num_samples]

        Returns:
            means: Mean tensor [num_classes, feature_dim]
            covs: Covariance tensor [num_classes, feature_dim, feature_dim]
                  (or [num_classes, feature_dim] if diagonal_cov=True)
            class_offsets: List of global class indices (for multi-task scenarios)
        """
        unique_labels = torch.unique(Y).sort().values
        num_classes = len(unique_labels)
        feature_dim = X.shape[1]

        means = torch.zeros((num_classes, feature_dim), device=self.device)
        if self.diagonal_cov:
            covs = torch.zeros((num_classes, feature_dim), device=self.device)
        else:
            covs = torch.zeros(
                (num_classes, feature_dim, feature_dim), device=self.device)

        for idx, label in enumerate(unique_labels):
            mask = (Y == label)
            class_samples = X[mask].float()

            # Mean
            means[idx] = class_samples.mean(dim=0)

            # Covariance
            if class_samples.shape[0] > 1:
                if self.diagonal_cov:
                    covs[idx] = class_samples.var(dim=0, unbiased=True)
                else:
                    covs[idx] = torch.cov(class_samples.T)
            else:
                # For single sample — zero covariance
                if self.diagonal_cov:
                    covs[idx] = torch.zeros(feature_dim, device=self.device)
                else:
                    covs[idx] = torch.zeros(
                        (feature_dim, feature_dim), device=self.device)

        # Global class indices
        class_offsets = list(range(num_classes))

        return means, covs, class_offsets

    def _bures_wasserstein_distance(
        self,
        mean1: torch.Tensor,
        cov1: torch.Tensor,
        mean2: torch.Tensor,
        cov2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Bures-Wasserstein distance between two Gaussian distributions.

        Formula:
            W₂²(𝒩₁, 𝒩₂) = ‖μ₁ - μ₂‖² + Tr(Σ₁ + Σ₂ - 2(Σ₁^{1/2} Σ₂ Σ₁^{1/2})^{1/2})

        Implementation based on OTDD (Microsoft Research):
        https://github.com/microsoft/otdd

        Args:
            mean1, mean2: Mean vectors [feature_dim]
            cov1, cov2: Covariance matrices [feature_dim, feature_dim]
                        or diagonals [feature_dim] if diagonal_cov=True

        Returns:
            d: Scalar Wasserstein distance (not squared)
        """
        # Distance between means
        d_mean = torch.sum((mean1 - mean2) ** 2)

        # Covariance term (Bures distance)
        bures_term = compute_bures_term(
            cov1, cov2,
            diagonal_cov=self.diagonal_cov,
            num_iters=self.sqrt_niters
        )

        # Final distance (not squared)
        w2_squared = d_mean + bures_term
        return torch.sqrt(torch.clamp(w2_squared, min=0.0))

    def _exact_wasserstein_distance(
        self,
        X1: torch.Tensor,
        X2: torch.Tensor
    ) -> float:
        """
        Compute exact Wasserstein-2 distance via EMD (Earth Mover's Distance).

        Args:
            X1: Tensor of points from first distribution [n_samples1, feature_dim]
            X2: Tensor of points from second distribution [n_samples2, feature_dim]

        Returns:
            d: Scalar Wasserstein-2 distance
        """
        C = ot.dist(X1.cpu().numpy(), X2.cpu().numpy(), metric='euclidean')
        a = ot.unif(X1.shape[0])
        b = ot.unif(X2.shape[0])
        w2_squared = ot.emd2(a, b, C, numItermax=1_000_000)
        return np.sqrt(w2_squared)

    def compute_pairwise_distances(
        self,
        datasets: List[Union[Dataset, DataLoader]],
        symmetric: bool = True
    ) -> torch.Tensor:
        """
        Compute pairwise distance matrix between all classes across all datasets.

        Args:
            datasets: List of datasets/dataloaders. Each must contain
                      integer labels in range [0, num_classes-1].
            symmetric: Flag for symmetric distances. If True, distance between
                       classes within the same dataset is computed once.

        Returns:
            D: Distance tensor of shape [total_classes, total_classes], where
               total_classes = sum(num_classes_per_dataset).
               D[i, j] represents distance between class i and class j in global numbering.
        """
        # Step 1: Collect statistics for each dataset
        dataset_stats: List[Tuple[torch.Tensor, torch.Tensor, List[int]]] = []
        class_offsets: List[int] = [0]

        for idx, dataset in enumerate(datasets):
            X, Y = self.preprocess_dataset(dataset, dataset_id=idx)

            if idx in self._stats_cache:
                means, covs, local_offsets = self._stats_cache[idx]
            else:
                means, covs, local_offsets = self._compute_gaussian_stats(X, Y)
                self._stats_cache[idx] = (means, covs, local_offsets)

            dataset_stats.append((means, covs, local_offsets))
            class_offsets.append(class_offsets[-1] + len(local_offsets))

        total_classes = class_offsets[-1]

        # Step 2: Compute distances
        if self.gaussian_assumption and self.diagonal_cov:
            # Vectorized path for diagonal Gaussian case — O(N²·d) via BLAS,
            # avoids Python loops over class pairs.
            all_means = torch.cat([s[0] for s in dataset_stats], dim=0)  # [N, d]
            all_vars  = torch.cat([s[1] for s in dataset_stats], dim=0)  # [N, d]

            # Mean term: ||mu_i - mu_j||²
            mean_sq = torch.cdist(all_means, all_means, p=2) ** 2  # [N, N]

            # Bures term: Σ(var_i + var_j - 2√(var_i·var_j))
            var_sums  = all_vars.sum(dim=1)                                   # [N]
            sqrt_vars = torch.sqrt(all_vars + 1e-12)                          # [N, d]
            cross     = sqrt_vars @ sqrt_vars.T                                # [N, N]
            bures_mat = var_sums.unsqueeze(1) + var_sums.unsqueeze(0) - 2 * cross  # [N, N]

            D = torch.sqrt(torch.clamp(mean_sq + bures_mat, min=0.0))
            if symmetric:
                D = (D + D.T) / 2
        else:
            D = torch.zeros((total_classes, total_classes), device=self.device)
            for i in range(len(datasets)):
                means_i, covs_i, offsets_i = dataset_stats[i]
                start_i = class_offsets[i]

                for j in range(i if symmetric else 0, len(datasets)):
                    means_j, covs_j, offsets_j = dataset_stats[j]
                    start_j = class_offsets[j]

                    for idx_i, local_i in enumerate(offsets_i):
                        global_i = start_i + idx_i
                        for idx_j, local_j in enumerate(offsets_j):
                            global_j = start_j + idx_j

                            if self.gaussian_assumption:
                                d = self._bures_wasserstein_distance(
                                    means_i[idx_i], covs_i[idx_i],
                                    means_j[idx_j], covs_j[idx_j]
                                )
                            else:
                                X_i, Y_i = self._data_cache.get(
                                    i, self.preprocess_dataset(datasets[i], dataset_id=i))
                                X_j, Y_j = self._data_cache.get(
                                    j, self.preprocess_dataset(datasets[j], dataset_id=j))
                                samples_i = X_i[Y_i == local_i]
                                samples_j = X_j[Y_j == local_j]
                                d = torch.tensor(
                                    self._exact_wasserstein_distance(samples_i, samples_j),
                                    device=self.device)

                            D[global_i, global_j] = d
                            if symmetric and i != j:
                                D[global_j, global_i] = d

        return D

    def embed_distance_matrix(
        self,
        distance_matrix: torch.Tensor,
        emb_dim: Optional[int] = None
    ) -> torch.Tensor:
        """
        Transform distance matrix into embeddings via Multidimensional Scaling (MDS).

        Args:
            distance_matrix: Distance tensor of shape [N, N], N = total_classes.
                             Must be symmetric with zero diagonal.
            emb_dim: Embedding dimensionality. If None, uses self.emb_dim.

        Returns:
            embeddings: Embedding tensor of shape [N, emb_dim]
        """
        target_dim = emb_dim if emb_dim is not None else self.emb_dim
        D_np = distance_matrix.cpu().numpy()

        np.fill_diagonal(D_np, 0.0)
        D_np = (D_np + D_np.T) / 2

        mds = MDS(
            n_components=target_dim,
            dissimilarity="precomputed",
            n_init=10,
            max_iter=10000,
            random_state=42
        )
        embeddings_np = mds.fit_transform(D_np)

        return torch.from_numpy(embeddings_np).to(self.device).float()

    def augment_features(
        self,
        data: Union[Dataset, DataLoader],
        label_embeddings: torch.Tensor,
        dataset_idx: int,
        class_offsets: List[int]
    ) -> torch.Tensor:
        """
        Augment original features with label embeddings for each sample.

        Args:
             Dataset or DataLoader to process.
            label_embeddings: Embeddings of all classes [total_classes, emb_dim].
            dataset_idx: Index of current dataset in the original datasets list.
            class_offsets: List of class offsets for each dataset.

        Returns:
            Z: Augmented feature tensor [num_samples, feature_dim + emb_dim]
        """
        X, Y = self.preprocess_dataset(data, dataset_id=dataset_idx)

        start_offset = class_offsets[dataset_idx]
        end_offset = class_offsets[dataset_idx + 1] if dataset_idx + \
            1 < len(class_offsets) else label_embeddings.shape[0]
        label_emb_for_dataset = label_embeddings[start_offset:end_offset]

        label_indices = Y.long()
        label_embs = label_emb_for_dataset[label_indices]

        Z = torch.cat([X, label_embs], dim=1)
        return Z

    def compute_wte(
        self,
        datasets: List[Union[Dataset, DataLoader]],
        reference: Optional[torch.Tensor] = None,
        create_reference: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Main method: compute Wasserstein Transport Embeddings for dataset collection.

        Args:
            datasets: List of datasets/dataloaders.
            reference: Optional reference distribution [ref_size, feature_dim + emb_dim].
            create_reference: If True and reference=None, creates reference from merged data.

        Returns:
            task_embeddings: [num_datasets, ref_size, feature_dim + emb_dim]
            label_embeddings: [total_classes, emb_dim]
            augmented_datasets: List of [num_samples, feature_dim + emb_dim]
        """
        D = self.compute_pairwise_distances(datasets, symmetric=True)
        label_embeddings = self.embed_distance_matrix(D, emb_dim=self.emb_dim)

        class_offsets = [0]
        for idx, dataset in enumerate(datasets):
            X, Y = self.preprocess_dataset(dataset, dataset_id=idx)
            num_classes = len(torch.unique(Y))
            class_offsets.append(class_offsets[-1] + num_classes)

        augmented_datasets: List[torch.Tensor] = []
        for idx, dataset in enumerate(datasets):
            Z = self.augment_features(
                dataset, label_embeddings, idx, class_offsets)
            augmented_datasets.append(Z)

        if reference is None and create_reference:
            all_data = torch.cat(augmented_datasets, dim=0)
            ref_size = min(1000, all_data.shape[0] // len(datasets))
            ref_indices = torch.randperm(all_data.shape[0])[:ref_size]
            reference = all_data[ref_indices].float()
        elif reference is None:
            raise ValueError(
                "Either provide 'reference' or set 'create_reference=True'")

        task_embeddings = []
        ref_size = reference.shape[0]

        for Z in augmented_datasets:
            Z = Z.float()
            C = ot.dist(Z.cpu().numpy(), reference.cpu().numpy(),
                        metric='euclidean')
            gamma = ot.emd(ot.unif(Z.shape[0]), ot.unif(
                ref_size), C, numItermax=1_000_000)
            gamma = torch.from_numpy(gamma).float().to(self.device)
            f = (ref_size * gamma.T @ Z - reference) / np.sqrt(ref_size)
            task_embeddings.append(f)

        task_embeddings_tensor = torch.stack(task_embeddings, dim=0)
        return task_embeddings_tensor, label_embeddings, augmented_datasets

    def embed(self, datasets, **kwargs):
        """Compute WTE embeddings — satisfies BaseEmbedder abstract interface."""
        return self.compute_wte(datasets, **kwargs)

    def clear_cache(self) -> None:
        """Clear all caches to free memory."""
        self._stats_cache.clear()
        self._data_cache.clear()
