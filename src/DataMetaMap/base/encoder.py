from abc import ABC, abstractmethod
from typing import Any

from torch import Tensor, nn


class BaseEncoder(ABC):
    """
    Base class for feature extraction backends.
    Отвечает за преобразование сырых данных в признаки
    перед подачей в эмбеддер.
    """

    @abstractmethod
    def encode(self, data: Any) -> Tensor:
        """
        Преобразует входные данные в тензор признаков.

        Args:
            data: Входные данные (изображения, таблицы и т.д.)

        Returns:
            Tensor: Закодированные признаки
        """
        pass
