from typing import Callable

import torch
from torch import Tensor, nn

from ..base.encoder import BaseEncoder


class PretrainedEncoder(BaseEncoder):
    """
    Encoder на основе предобученной модели (например ResNet).
    Замораживает веса и использует модель только для inference.
    """

    def __init__(self, model: nn.Module, transform: Callable | None = None):
        """ 
        Args:
            model: Предобученная PyTorch модель
            transform: Опциональные трансформации входных данных
        """
        self.model = model
        self.transform = transform

        # Замораживаем веса
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    def encode(self, data: Tensor) -> Tensor:
        if self.transform is not None:
            data = self.transform(data)
        with torch.no_grad():
            return self.model(data)
