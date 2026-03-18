from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class BaseMetric(ABC):
    """
    Base class for similarity and distance metrics.
    """

    @abstractmethod
    def compute(
        self,
        embedding1: NDArray[np.floating],
        embedding2: NDArray[np.floating],
    ) -> float:
        """
        Вычисляет метрику между двумя эмбеддингами.

        Args:
            embedding1: Первый вектор
            embedding2: Второй вектор

        Returns:
            float: Значение метрики
        """
        pass

    def compute_matrix(
        self,
        embeddings: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """
        Вычисляет матрицу попарных метрик.

        Args:
            embeddings: Матрица эмбеддингов (n_datasets, embedding_dim)

        Returns:
            NDArray: Матрица метрик (n_datasets, n_datasets)
        """
        n = len(embeddings)
        matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                matrix[i, j] = self.compute(embeddings[i], embeddings[j])
        return matrix