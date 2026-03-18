import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from torch import Tensor

from ..base.encoder import BaseEncoder


class IdentityEncoder(BaseEncoder):
    """
    Encoder который возвращает данные как есть,
    только конвертируя в Tensor если нужно.
    """

    def encode(self, data: Tensor | NDArray | pd.DataFrame) -> Tensor:
        if isinstance(data, Tensor):
            return data.float()
        elif isinstance(data, np.ndarray):
            return torch.from_numpy(data).float()
        elif isinstance(data, pd.DataFrame):
            return torch.from_numpy(data.values).float()
        else:
            raise TypeError(f"Unsupported type: {type(data)}")
