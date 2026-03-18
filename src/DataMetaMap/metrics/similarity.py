import numpy as np
from numpy.typing import NDArray


class CosineSimilarity:
    """Cosine similarity between two vectors."""

    def compute(
        self,
        embedding1: NDArray[np.floating],
        embedding2: NDArray[np.floating],
    ) -> float:
        dot = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(dot / (norm1 * norm2))


class EuclideanDistance:
    """Euclidean distance between two vectors."""

    def compute(
        self,
        embedding1: NDArray[np.floating],
        embedding2: NDArray[np.floating],
    ) -> float:
        return float(np.linalg.norm(embedding1 - embedding2))
