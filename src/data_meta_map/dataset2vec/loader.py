from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from torch import Tensor, from_numpy

from .utils import (
    DataUtils,
    InconsistentTypesException,
    InvalidDataTypeException,
)


class Dataset2VecLoader:
    """Data loader for Dataset2Vec training.

    Each iteration yields a batch of (X1, y1, X2, y2, label) tuples,
    where label=1 if both samples are drawn from the same dataset
    (positive pair) and label=0 otherwise (negative pair).
    """

    def __init__(
        self,
        batch_size: int = 32,
        n_batches: int = 100,
    ):
        self.batch_size = batch_size
        self.n_batches = n_batches
        self.Xs: list[Tensor] = []
        self.ys: list[Tensor] = []
        self.n_datasets: int = 0
        self.released_batches_count: int = 0

    def load(
        self,
        data: (
            Path
            | list[Path]
            | list[pd.DataFrame]
            | list[NDArray[np.generic]]
            | list[Tensor]
        ),
    ) -> "Dataset2VecLoader":
        datasets = self._read_data_if_needed(data)
        self.n_datasets = len(datasets)
        self.released_batches_count = 0
        self._setup_xs(datasets)
        self._setup_ys(datasets)
        return self

    # ------------------------------------------------------------------ #
    #                         Data preparation                            #
    # ------------------------------------------------------------------ #

    def _setup_xs(self, datasets) -> None:
        Xs = [
            self._normalize_to_pandas(ds).iloc[:, :-1]
            for ds in datasets
        ]
        Xs = [
            DataUtils.get_preprocessing_pipeline()
            .fit_transform(X).values
            for X in Xs
        ]
        self.Xs = [self._to_torch(X) for X in Xs]

    def _setup_ys(self, datasets) -> None:
        ys = [
            self._normalize_to_pandas(ds).iloc[:, -1]
            for ds in datasets
        ]
        self.ys = [
            self._to_torch(y.values).reshape(-1, 1)
            for y in ys
        ]

    def _read_data_if_needed(self, data):
        if isinstance(data, Path):
            return [
                pd.read_csv(f) for f in sorted(data.iterdir())
            ]
        elif (
            isinstance(data, list)
            and len(data) > 0
            and isinstance(data[0], Path)
        ):
            if any(not isinstance(el, Path) for el in data):
                raise InconsistentTypesException(
                    "If any element is Path — all must be Path"
                )
            return [pd.read_csv(f) for f in data]
        return data

    def _to_torch(self, data: NDArray[np.generic]) -> Tensor:
        if isinstance(data, np.ndarray):
            return from_numpy(data).type(torch.float)
        raise InvalidDataTypeException(
            f"{type(data)} is not NDArray."
        )

    def _normalize_to_pandas(self, data) -> pd.DataFrame:
        if isinstance(data, Tensor):
            return pd.DataFrame(data.numpy())
        elif isinstance(data, pd.DataFrame):
            return data
        elif isinstance(data, np.ndarray):
            return pd.DataFrame(data)
        raise InvalidDataTypeException(
            f"{type(data)} is not supported. "
            "Use Tensor, DataFrame or NDArray."
        )

    # ------------------------------------------------------------------ #
    #                          Iterator protocol                          #
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return self.n_batches

    def __iter__(self) -> Dataset2VecLoader:
        return deepcopy(self)

    def __next__(
        self,
    ) -> list[tuple[Tensor, Tensor, Tensor, Tensor, int]]:
        if self.released_batches_count == self.n_batches:
            raise StopIteration()
        self.released_batches_count += 1
        return self._get_batch()

    # ------------------------------------------------------------------ #
    #                          Batch generation                           #
    # ------------------------------------------------------------------ #

    def _get_batch(
        self,
    ) -> list[tuple[Tensor, Tensor, Tensor, Tensor, int]]:
        return [
            self._get_single_example()
            for _ in range(self.batch_size)
        ]

    def _get_single_example(
        self,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, int]:
        idx1, idx2 = self._get_random_dataset_indices()
        return (
            *self._get_subsample(idx1),
            *self._get_subsample(idx2),
            int(idx1 == idx2),
        )

    def _get_random_dataset_indices(self) -> tuple[int, int]:
        """Sample two dataset indices — same dataset with probability 0.5 (positive pair)."""
        if np.random.uniform() >= 0.5:
            idx = np.random.choice(self.n_datasets)
            return (idx, idx)
        idx1, idx2 = np.random.choice(
            self.n_datasets, 2, replace=False
        ).astype(int)
        return (idx1, idx2)

    def _get_subsample(
        self, dataset_idx: int
    ) -> tuple[Tensor, Tensor]:
        X = self.Xs[dataset_idx]
        y = self.ys[dataset_idx]
        rows_idx, feat_idx, tgt_idx = self._sample_indices(X, y)
        X = DataUtils.index_tensor_using_lists(X, rows_idx, feat_idx)
        y = DataUtils.index_tensor_using_lists(y, rows_idx, tgt_idx)
        return X, y

    def _sample_indices(
        self, X: Tensor, y: Tensor
    ) -> tuple[NDArray, NDArray, NDArray]:
        """
        Sample row indices (power-of-two count between 8 and 256),
        a random feature subset, and a random target subset.
        """
        n_rows = X.shape[0]
        assert n_rows >= 8, "Dataset must have at least 8 rows"

        max_q = min(int(np.log2(n_rows)), 8)
        q = np.random.choice(np.arange(3, max_q + 1))
        rows_idx = np.random.choice(n_rows, 2 ** q)
        feat_idx = DataUtils.sample_random_subset(X.shape[1])
        tgt_idx = DataUtils.sample_random_subset(y.shape[1])
        return rows_idx, feat_idx, tgt_idx


class RepeatableDataset2VecLoader(Dataset2VecLoader):
    """
    Dataset2VecLoader variant that returns identical batches on every iter() call.
    Intended for validation and testing where determinism is required.
    """

    def load(self, data):
        super().load(data)
        get_batch = super()._get_batch
        self._fixed_batches = [
            get_batch()
            for _ in range(self.n_batches)
        ]
        return self

    def __iter__(self) -> "RepeatableDataset2VecLoader":
        return deepcopy(self)

    def __next__(self):
        if self.released_batches_count == len(self._fixed_batches):
            raise StopIteration()
        batch = self._fixed_batches[self.released_batches_count]
        self.released_batches_count += 1
        return batch
