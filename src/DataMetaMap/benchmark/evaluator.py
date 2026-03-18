from typing import Any


class Evaluator:
    """Simple evaluator wrapper for running benchmarks."""

    def __init__(self, embedder: Any, metric: Any):
        self.embedder = embedder
        self.metric = metric

    def evaluate(self, datasets: list[Any]) -> dict[str, float]:
        """Compute pairwise metric scores for a list of datasets."""
        embeddings = [self.embedder.embed(*d) for d in datasets]
        n = len(embeddings)
        scores = {}
        for i in range(n):
            for j in range(i + 1, n):
                key = f"{i}-{j}"
                scores[key] = self.metric.compute(embeddings[i], embeddings[j])
        return scores
