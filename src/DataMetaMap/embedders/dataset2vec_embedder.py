from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from numpy.typing import NDArray
from torch import Tensor

from ..base.embedder import BaseEmbedder
from .dataset2vec.config import Dataset2VecConfig, OptimizerConfig
from .dataset2vec.loader import (
    Dataset2VecLoader,
    RepeatableDataset2VecLoader,
)
from .dataset2vec.model import Dataset2Vec


class Dataset2VecEmbedder(BaseEmbedder):
    """
    Адаптер Dataset2Vec под интерфейс BaseEmbedder.

    Оборачивает Dataset2Vec модель и предоставляет
    унифицированный интерфейс embed() / fit().

    Args:
        config: Конфигурация архитектуры Dataset2Vec
        optimizer_config: Конфигурация оптимизатора
        max_epochs: Количество эпох обучения
        batch_size: Размер батча
        n_batches: Количество батчей на эпоху
    """

    def __init__(
        self,
        config: Dataset2VecConfig = Dataset2VecConfig(),
        optimizer_config: OptimizerConfig = OptimizerConfig(),
        max_epochs: int = 10,
        batch_size: int = 32,
        n_batches: int = 100,
    ):
        self.config = config
        self.optimizer_config = optimizer_config
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.n_batches = n_batches

        self._model = Dataset2Vec(config, optimizer_config)
        self._is_fitted = False

    def fit(
        self,
        data: Path | list[Path] | list[pd.DataFrame] | list[NDArray],
        val_data: Path | list[Path] | list[pd.DataFrame] | list[NDArray] | None = None,
        trainer_kwargs: dict | None = None,
    ) -> "Dataset2VecEmbedder":
        """Обучает Dataset2Vec на наборе датасетов.

        Args:
            data: Обучающие данные
            val_data: Валидационные данные (опционально)
            trainer_kwargs: Дополнительные аргументы для pytorch_lightning.Trainer

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
        self.trainer = pl.Trainer(max_epochs=self.max_epochs, **trainer_kwargs)
        self.trainer.fit(self._model, train_loader, val_loader)

        self._is_fitted = True
        return self

    def embed(
        self,
        X: Tensor,
        y: Tensor,
    ) -> NDArray[np.floating]:
        """Вычисляет эмбеддинг для одного датасета."""
        self._model.eval()
        embedding = self._model(X, y)
        return embedding.detach().cpu().numpy()

    def save(self, path: str) -> None:
        """Сохраняет веса модели."""
        import torch

        torch.save(self._model.state_dict(), path)

    def load(self, path: str) -> "Dataset2VecEmbedder":
        """Загружает веса модели."""
        import torch

        self._model.load_state_dict(torch.load(path))
        self._is_fitted = True
        return self
