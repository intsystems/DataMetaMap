from abc import ABC, abstractmethod
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
    Union,
    runtime_checkable
)
import torch
from torch.utils.data import Dataset, DataLoader


@runtime_checkable
class SupportsGetItem(Protocol):
    """
    Protocol for objects with __getitem__ and __len__ interface.
    Ensures compatibility with both Dataset and custom iterable objects.
    """
    def __getitem__(self, index: int) -> Tuple[Any, int]: ...
    def __len__(self) -> int: ...


class BaseEmbedder(ABC):
    """
    Abstract base class for dataset embedding.

    Designed to unify the transformation of arbitrary datasets
    (images, text, tabular data) into feature space with subsequent
    computation of geometric distances between distributions.

    Key Features:
        - Complete data-type agnosticism (works with any vectorized features)
        - Clear separation of stages: preprocessing → distances → embeddings → augmentation
        - Support for both raw datasets and pre-configured DataLoaders
        - Explicit type annotations for static analysis

    Implementation Requirements:
        Subclasses must implement all abstract methods with signatures matching
        the specified input/output types.
    """

    def __init__(
        self,
        emb_dim: int,
        device: Union[str, torch.device] = "cpu",
        max_samples: Optional[int] = None,
        batch_size: int = 64
    ):
        """
        Initialize base embedder.

        Args:
            emb_dim: Target dimensionality of label embeddings.
            device: Computation device ('cpu', 'cuda', or torch.device).
            max_samples: Maximum number of samples to process from dataset.
                         If None, all samples are used.
            batch_size: Batch size for DataLoader during data loading.
        """
        self.emb_dim = emb_dim
        self.device = torch.device(device)
        self.max_samples = max_samples
        self.batch_size = batch_size

    @abstractmethod
    def preprocess_dataset(
        self,
        data: Union[SupportsGetItem, DataLoader]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Transform dataset/dataloader into feature-label tensor pair.

        Args:
            data: Object supporting either:
                  - Dataset interface: must return (features, label) in __getitem__
                  - DataLoader interface: must yield batches (X_batch, y_batch)

                  Features must be convertible to tensor of shape [batch_size, feature_dim].
                  Labels must be integers (not one-hot encoded).

        Returns:
            X: Feature tensor of shape [num_samples, feature_dim], dtype=torch.float32
            Y: Label tensor of shape [num_samples], dtype=torch.long

        Implementation Requirements:
            - Automatically create DataLoader when Dataset is provided
            - Process data on self.device
            - Support self.max_samples parameter (subsampling without replacement)
            - Flatten features to [N, D] format if necessary
        """
        pass

    @abstractmethod
    def compute_pairwise_distances(
        self,
        datasets: List[Union[SupportsGetItem, DataLoader]],
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

        Global Class Numbering:
            Dataset 0: classes [0, ..., k₀-1]
            Dataset 1: classes [k₀, ..., k₀+k₁-1]
            ...

        Implementation Requirements:
            - Support different number of classes across datasets
            - Distance must satisfy: D[i, i] = 0, D[i, j] >= 0
            - If symmetric=True: D[i, j] = D[j, i]
        """
        pass

    @abstractmethod
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
            embeddings: Embedding tensor of shape [N, emb_dim], where each row
                        represents the embedding of the corresponding class
                        in global numbering.
        """
        pass

    @abstractmethod
    def augment_features(
        self,
        data: Union[SupportsGetItem, DataLoader],
        label_embeddings: torch.Tensor,
        dataset_idx: int,
        class_offsets: List[int]
    ) -> torch.Tensor:
        """
        Augment original features with label embeddings for each sample.

        Args:
            data: Dataset or DataLoader to process.
            label_embeddings: Embeddings of all classes [total_classes, emb_dim].
            dataset_idx: Index of current dataset in the original datasets list.
            class_offsets: List of class offsets for each dataset.
                           Example: [0, 10, 25] means dataset 0 has classes 0-9,
                           dataset 1 has classes 10-24, dataset 2 has classes 25+.

        Returns:
            Z: Augmented feature tensor of shape [num_samples, feature_dim + emb_dim],
               where label embedding is concatenated to each original feature vector.
        """
        pass

    def get_class_statistics(
        self,
        X: torch.Tensor,
        Y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute class-wise statistics (mean, covariance).

        Args:
            X: Feature tensor [num_samples, feature_dim]
            Y: Label tensor [num_samples], dtype=torch.long

        Returns:
            means: Mean tensor [num_classes, feature_dim]
            covs: Covariance tensor [num_classes, feature_dim, feature_dim]
        """
        unique_labels = torch.unique(Y)
        num_classes = len(unique_labels)
        feature_dim = X.shape[1]

        means = torch.zeros((num_classes, feature_dim), device=self.device)
        covs = torch.zeros((num_classes, feature_dim, feature_dim), device=self.device)

        for idx, label in enumerate(unique_labels):
            mask = (Y == label)
            class_samples = X[mask].float()
            means[idx] = class_samples.mean(dim=0)
            if class_samples.shape[0] > 1:
                covs[idx] = torch.cov(class_samples.T)
            else:
                # For single sample, covariance is undefined — use zero matrix
                covs[idx] = torch.zeros((feature_dim, feature_dim), device=self.device)

        return means, covs

    @property
    def device(self) -> torch.device:
        """Get current computation device."""
        return self._device

    @device.setter
    def device(self, device: Union[str, torch.device]) -> None:
        """Set computation device with validation."""
        self._device = torch.device(device)
