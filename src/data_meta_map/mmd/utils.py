"""Loader / dataset helpers shared across the MMD subpackage."""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler


def dataset_to_tensor(
    data: Union[Dataset, DataLoader],
    *,
    batch_size: int = 64,
    max_samples: Optional[int] = None,
    device: Union[str, torch.device] = "cpu",
    return_labels: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Materialize a ``Dataset`` / ``DataLoader`` into ``(X, y)`` tensors.

    Mirrors the behaviour of
    :meth:`data_meta_map.wasserstein_embedder.WassersteinEmbedder.preprocess_dataset`
    so the two embedders consume data in the same way.

    Args:
        data: Dataset (returns ``(features, label)`` pairs) or DataLoader.
        batch_size: Batch size when constructing a loader from a Dataset.
        max_samples: Optional subsampling cap (without replacement).
        device: Target device for the returned tensors.
        return_labels: If ``False``, returns ``(X, None)`` and does not
            try to read labels (useful for unlabeled tensors).

    Returns:
        Tuple ``(X, y)`` where ``X`` has shape ``[N, D]`` and ``y``
        has shape ``[N]`` (long dtype) -- or ``None`` if labels are
        not requested.
    """
    if isinstance(data, Dataset):
        if max_samples and len(data) > max_samples:
            idxs = np.sort(
                np.random.choice(len(data), max_samples, replace=False)
            )
            sampler = SubsetRandomSampler(idxs)
            loader = DataLoader(data, sampler=sampler, batch_size=batch_size)
        else:
            loader = DataLoader(data, batch_size=batch_size, shuffle=False)
    elif isinstance(data, DataLoader):
        loader = data
    else:
        raise TypeError(
            f"Expected Dataset or DataLoader, got {type(data).__name__}"
        )

    device = torch.device(device)
    X_list: List[torch.Tensor] = []
    y_list: List[torch.Tensor] = []

    for batch in loader:
        x_batch = batch[0]
        x_flat = x_batch.view(x_batch.size(0), -1).float().to(device)
        X_list.append(x_flat)
        if return_labels and len(batch) > 1:
            y_list.append(batch[1].long().view(-1).to(device))

    X = torch.cat(X_list, dim=0)
    y = torch.cat(y_list, dim=0) if y_list else None
    return X, y


def split_by_class(
    X: torch.Tensor, y: torch.Tensor
) -> List[torch.Tensor]:
    """Return the list of per-class feature subsets in label order."""
    classes = torch.unique(y).sort().values
    return [X[y == c] for c in classes]
