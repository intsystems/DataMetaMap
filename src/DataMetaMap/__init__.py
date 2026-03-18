from .embedders import Dataset2VecEmbedder
from .embedders.dataset2vec.config import Dataset2VecConfig, OptimizerConfig
from .base.embedder import BaseEmbedder
from .base.metric import BaseMetric
from .base.encoder import BaseEncoder
from .encoders.identity import IdentityEncoder
from .encoders.pretrained import PretrainedEncoder
from .encoders.trainable import TrainableEncoder

__version__ = '0.0.1'

__all__ = [
    # Base classes
    "BaseEmbedder",
    "BaseMetric",
    "BaseEncoder",

    # Embedders
    "Dataset2VecEmbedder",

    # Encoders
    "IdentityEncoder",
    "PretrainedEncoder",
    "TrainableEncoder",

    # Configs
    "Dataset2VecConfig",
    "OptimizerConfig",
]
