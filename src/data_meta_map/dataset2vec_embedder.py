from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from numpy.typing import NDArray
from torch import Tensor

from data_meta_map.base_embedder import BaseEmbedder
from data_meta_map.dataset2vec.model import Dataset2Vec
from data_meta_map.dataset2vec.loader import (
    Dataset2VecLoader,
    RepeatableDataset2VecLoader,
)


def dataset2vec(
    model: Dataset2Vec,
    X: Tensor,
    y: Tensor,
    fit_data: Path | list[Path] | list[pd.DataFrame] | list | None = None,
    **kwargs,
) -> NDArray[np.floating]:
    """Convenience function: fit (optional) and embed a single tabular dataset.

    Consistent with the task2vec() convenience function in task2vec.py.

    Args:
        model: Pre-initialized Dataset2Vec model (e.g. from get_model('dataset2vec')).
        X: Feature tensor [n_samples, n_features].
        y: Target tensor [n_samples].
        fit_data: Training datasets to fit the model before embedding.
                  If None, the model is used as-is (must already be trained).
        **kwargs: Forwarded to Dataset2VecEmbedder.__init__.

    Returns:
        NDArray: Embedding vector of shape [output_size].
    """
    embedder = Dataset2VecEmbedder(model, **kwargs)
    if fit_data is not None:
        embedder.fit(fit_data)
    return embedder.embed(X, y)


class Dataset2VecEmbedder(BaseEmbedder):
    """
    Dataset embedder based on Dataset2Vec (Iwata & Ghahramani, 2020).

    Consistent with Task2Vec and WassersteinEmbedder: the model is injected
    via __init__ rather than created internally.

    Typical usage:
        model = get_model('dataset2vec')
        embedder = Dataset2VecEmbedder(model, max_epochs=20)
        embedder.fit(train_datasets)
        embedding = embedder.embed(X_test, y_test)

    Args:
        model: Pre-initialized Dataset2Vec model.
        max_epochs: Number of training epochs.
        batch_size: Batch size for the data loader.
        n_batches: Number of batches per epoch.
    """

    def __init__(
        self,
        model: Dataset2Vec,
        max_epochs: int = 10,
        batch_size: int = 32,
        n_batches: int = 100,
    ):
        self.model = model
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.n_batches = n_batches
        self._is_fitted = False

    def fit(
        self,
        data: Path | list[Path] | list[pd.DataFrame] | list,
        val_data: Path | list[Path] | list[pd.DataFrame] | list | None = None,
        trainer_kwargs: dict | None = None,
    ) -> "Dataset2VecEmbedder":
        """Train Dataset2Vec on a collection of tabular datasets.

        Args:
            data: Training data. Accepts a directory path, a list of file
                  paths, DataFrames, or NDArrays. Each element is one dataset;
                  the last column is treated as the target.
            val_data: Validation data in the same format. Optional.
            trainer_kwargs: Additional kwargs for pytorch_lightning.Trainer.

        Returns:
            self
        """
        train_loader = Dataset2VecLoader(
            batch_size=self.batch_size,
            n_batches=self.n_batches,
        ).load(data)

        val_loader = None
        if val_data is not None:
            val_loader = RepeatableDataset2VecLoader(
                batch_size=self.batch_size,
                n_batches=self.n_batches // 5,
            ).load(val_data)

        trainer_kwargs = trainer_kwargs or {}
        trainer = pl.Trainer(max_epochs=self.max_epochs, **trainer_kwargs)
        trainer.fit(self.model, train_loader, val_loader)

        self._is_fitted = True
        return self

    def embed(
        self,
        X: Tensor,
        y: Tensor,
    ) -> NDArray[np.floating]:
        """Compute embedding for a single tabular dataset.

        Args:
            X: Feature tensor of shape [n_samples, n_features].
            y: Target tensor of shape [n_samples] or [n_samples, 1].

        Returns:
            NDArray: 1-D embedding vector of shape [output_size].

        Raises:
            RuntimeError: If called before fit().
        """
        if not self._is_fitted:
            raise RuntimeError(
                "Model is not fitted. Call fit() before embed()."
            )
        self.model.eval()
        embedding = self.model(X, y)
        return embedding.detach().cpu().numpy()

    def save(self, path: str) -> None:
        """Save model weights to disk.

        Args:
            path: File path for the saved state dict.
        """
        import torch
        torch.save(self.model.state_dict(), path)

    def load(self, path: str) -> "Dataset2VecEmbedder":
        """Load model weights from disk.

        Args:
            path: File path to the saved state dict.

        Returns:
            self
        """
        import torch
        self.model.load_state_dict(torch.load(path))
        self._is_fitted = True
        return self
