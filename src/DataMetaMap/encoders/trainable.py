from torch import Tensor, nn

from ..base.encoder import BaseEncoder


class TrainableEncoder(BaseEncoder):
    """
    Encoder с обучаемыми весами.
    Dataset2Vec model используется именно как TrainableEncoder.
    """

    def __init__(self, model: nn.Module):
        """
        Args:
            model: Обучаемая PyTorch модель (например Dataset2Vec)
        """
        self.model = model

    def encode(self, *args, **kwargs) -> Tensor:
        return self.model(*args, **kwargs)

    def train_mode(self) -> "TrainableEncoder":
        self.model.train()
        return self

    def eval_mode(self) -> "TrainableEncoder":
        self.model.eval()
        return self
