from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from torch import Tensor


class BaseDatasetLoader(ABC):
    """
    Базовый класс для всех загрузчиков данных.
    Каждый эмбеддер использует свой тип загрузчика:
        - Dataset2Vec  → TabularLoader
        - Task2Vec     → ImageLoader
        - Wasserstein  → WassersteinLoader
    """

    def __init__(
        self,
        batch_size: int = 32,
        n_batches: int = 100,
    ):
        self.batch_size = batch_size
        self.n_batches = n_batches

    @abstractmethod
    def load(
        self,
        data: Path | list[Path] | list[pd.DataFrame] | list[NDArray],
    ) -> "BaseDatasetLoader":
        """
        Загружает и подготавливает данные.

        Args:
            data: Входные данные — пути, датафреймы или массивы

        Returns:
            self — для цепочки вызовов: loader.load(data).iterate()
        """
        pass

    @abstractmethod
    def __iter__(self) -> Iterator:
        """Возвращает итератор по батчам."""
        pass

    @abstractmethod
    def __next__(self):
        """Возвращает следующий батч."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Возвращает количество батчей."""
        pass