from data_meta_map.base_embedder import BaseEmbedder
from data_meta_map.wasserstein_embedder import WassersteinEmbedder
try:
    from data_meta_map.dataset2vec_embedder import Dataset2VecEmbedder, dataset2vec
    from data_meta_map.mmd_embedder import MMDEmbedder, mmd
except Exception:  # pragma: no cover
    # Allow importing the package even when optional heavy deps (e.g. lightning)
    # are not available or are broken in the current environment.
    Dataset2VecEmbedder = None  # type: ignore
    dataset2vec = None  # type: ignore
    MMDEmbedder = None
    mmd = None
from data_meta_map.dataset2vec_embedder import Dataset2VecEmbedder, dataset2vec
from data_meta_map.mmd_embedder import MMDEmbedder, mmd

__all__ = [
    "BaseEmbedder",
    "WassersteinEmbedder",
    "Dataset2VecEmbedder",
    "dataset2vec",
    "MMDEmbedder",
    "mmd",
]

if Dataset2VecEmbedder is not None:
    __all__ += ["Dataset2VecEmbedder", "dataset2vec"]
