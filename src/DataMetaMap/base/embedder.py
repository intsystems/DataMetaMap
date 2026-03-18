from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import NDArray


class BaseEmbedder(ABC):
    """
    Unified interface for all embedding methods.
    Все эмбеддеры (Task2Vec, Dataset2Vec, Wasserstein)
    должны наследоваться от этого класса.
    """

    @abstractmethod
    def embed(self, *args, **kwargs) -> NDArray[np.floating]:
        """
        Вычисляет векторное представление датасета.

        Returns:
            NDArray: Embedding vector
        """
        pass

    @abstractmethod
    def fit(self, *args, **kwargs) -> "BaseEmbedder":
        """
        Обучает эмбеддер если требуется.
        Для предобученных моделей — no-op.

        Returns:
            self
        """
        pass

    def fit_embed(self, *args, **kwargs) -> NDArray[np.floating]:
        """
        Удобный метод: обучение + получение эмбеддинга за один вызов.
        """
        self.fit(*args, **kwargs)
        return self.embed(*args, **kwargs)