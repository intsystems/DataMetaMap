import numpy as np
import pandas as pd
import pytest
import torch

from data_meta_map.base_embedder import BaseEmbedderDEPRECATED
from data_meta_map.dataset2vec.loader import (
    Dataset2VecLoader,
    RepeatableDataset2VecLoader,
)
from data_meta_map.dataset2vec.train import LightningBase
from data_meta_map.dataset2vec.utils import (
    DataUtils,
    InconsistentTypesException,
    InvalidDataTypeException,
    Validators,
)


def _make_tabular_np(n_rows: int = 32, n_features: int = 5) -> np.ndarray:
    # Last column is treated as target by Dataset2VecLoader.
    X = np.random.RandomState(0).randn(n_rows, n_features).astype(np.float32)
    y = (np.arange(n_rows) % 2).astype(np.float32).reshape(-1, 1)
    return np.concatenate([X, y], axis=1)


# -----------------------------------------------------------------------------
# utils.py
# -----------------------------------------------------------------------------


def test_validators_smoke():
    assert Validators.is_positive(1) == 1
    assert Validators.non_negative(0) == 0
    assert Validators.all_elements_positive([1, 2]) == [1, 2]
    assert Validators.non_empty([1]) == [1]


def test_sample_random_subset_with_int_and_singleton(monkeypatch):
    # int input -> becomes arange(a)
    monkeypatch.setattr(np.random, "uniform", lambda *args, **kwargs: np.zeros(5))
    subset = DataUtils.sample_random_subset(5)
    assert np.array_equal(subset, np.arange(5))

    # singleton array must return itself (no randomness)
    subset2 = DataUtils.sample_random_subset(np.array([7]))
    assert np.array_equal(subset2, np.array([7]))


def test_sample_random_subset_returns_all_when_empty_subset(monkeypatch):
    # Force "all False" mask: uniform returns 1.0 -> all comparisons (<0.5) are False.
    monkeypatch.setattr(np.random, "uniform", lambda *args, **kwargs: np.ones(4))
    a = np.arange(4)
    subset = DataUtils.sample_random_subset(a)
    assert np.array_equal(subset, a)


def test_index_tensor_using_lists():
    t = torch.arange(20).reshape(5, 4)
    rows = np.array([0, 2, 4])
    cols = np.array([1, 3])
    out = DataUtils.index_tensor_using_lists(t, rows, cols)
    assert out.shape == (3, 2)
    assert torch.equal(out, t[rows][:, cols])


# -----------------------------------------------------------------------------
# loader.py
# -----------------------------------------------------------------------------


def test_loader_read_data_from_directory(tmp_path):
    df1 = pd.DataFrame(_make_tabular_np(16, 3))
    df2 = pd.DataFrame(_make_tabular_np(20, 3))
    (tmp_path / "a.csv").write_text(df1.to_csv(index=False))
    (tmp_path / "b.csv").write_text(df2.to_csv(index=False))

    loader = Dataset2VecLoader(batch_size=2, n_batches=1).load(tmp_path)
    assert loader.n_datasets == 2
    assert len(loader.Xs) == 2
    assert len(loader.ys) == 2
    assert loader.Xs[0].ndim == 2
    assert loader.ys[0].ndim == 2


def test_loader_read_data_list_of_paths_inconsistent_types_raises(tmp_path):
    p = tmp_path / "a.csv"
    p.write_text(pd.DataFrame(_make_tabular_np(16, 3)).to_csv(index=False))

    with pytest.raises(InconsistentTypesException):
        Dataset2VecLoader().load([p, pd.DataFrame(_make_tabular_np(16, 3))])


def test_loader_to_torch_raises_on_invalid_type():
    loader = Dataset2VecLoader()
    with pytest.raises(InvalidDataTypeException):
        loader._to_torch([1, 2, 3])  # type: ignore[arg-type]


def test_loader_normalize_to_pandas_supported_types():
    loader = Dataset2VecLoader()
    t = torch.randn(8, 3)
    df = pd.DataFrame(np.random.randn(8, 3))
    arr = np.random.randn(8, 3)
    assert isinstance(loader._normalize_to_pandas(t), pd.DataFrame)
    assert isinstance(loader._normalize_to_pandas(df), pd.DataFrame)
    assert isinstance(loader._normalize_to_pandas(arr), pd.DataFrame)


def test_loader_normalize_to_pandas_invalid_type_raises():
    loader = Dataset2VecLoader()
    with pytest.raises(InvalidDataTypeException):
        loader._normalize_to_pandas("nope")  # type: ignore[arg-type]


def test_loader_iter_and_stopiteration():
    data = [pd.DataFrame(_make_tabular_np(32, 4)), pd.DataFrame(_make_tabular_np(40, 4))]
    loader = Dataset2VecLoader(batch_size=3, n_batches=2).load(data)

    it = iter(loader)
    batch1 = next(it)
    assert isinstance(batch1, list)
    assert len(batch1) == 3
    assert len(batch1[0]) == 5

    batch2 = next(it)
    assert len(batch2) == 3

    with pytest.raises(StopIteration):
        next(it)


def test_repeatable_loader_returns_same_batches_each_iter():
    data = [pd.DataFrame(_make_tabular_np(32, 4)), pd.DataFrame(_make_tabular_np(40, 4))]
    loader = RepeatableDataset2VecLoader(batch_size=2, n_batches=2).load(data)

    it1 = iter(loader)
    it2 = iter(loader)
    b11 = next(it1)
    b21 = next(it2)

    # Compare tensor values inside the first example of the batch.
    for i in range(4):
        assert torch.equal(b11[0][i], b21[0][i])
    assert b11[0][4] == b21[0][4]


# -----------------------------------------------------------------------------
# train.py (LightningBase)
# -----------------------------------------------------------------------------


class _ToyLightning(LightningBase):
    def forward(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Simple deterministic embedding.
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        return torch.cat([X.mean(dim=0, keepdim=True), y.mean(dim=0, keepdim=True)], dim=1).squeeze(0)

    def calculate_loss(self, labels: torch.Tensor, similarities: torch.Tensor) -> torch.Tensor:
        # Encourage similarities to match labels.
        labels = labels.float()
        return torch.mean((similarities - labels) ** 2)


def _make_batch(batch_size: int = 4) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]]:
    batch = []
    for i in range(batch_size):
        X1 = torch.randn(16, 5)
        y1 = torch.randint(0, 2, (16,)).float()
        X2 = torch.randn(16, 5)
        y2 = torch.randint(0, 2, (16,)).float()
        label = int(i % 2 == 0)
        batch.append((X1, y1, X2, y2, label))
    return batch


def test_extract_labels_and_similarities_shapes():
    m = _ToyLightning()
    batch = _make_batch(3)
    labels, sims = m.extract_labels_and_similarities_from_batch(batch)
    assert labels.shape == (3,)
    assert sims.shape == (3,)
    assert torch.all((sims >= 0) & (sims <= 1))


def test_training_step_and_epoch_hooks_smoke():
    m = _ToyLightning()
    batch = _make_batch(5)

    m.on_train_epoch_start()
    out = m.training_step(batch, batch_idx=0)
    assert "loss" in out and "predictions" in out

    m.on_train_batch_end(out, batch, batch_idx=0)
    assert len(m.training_predictions) == 1
    assert len(m.training_labels) == 1

    # Should not crash even without Lightning installed (log is a no-op in fallback).
    m.on_train_epoch_end()


def test_on_train_batch_end_rejects_non_mapping():
    m = _ToyLightning()
    m.on_train_epoch_start()
    batch = _make_batch(2)
    with pytest.raises(TypeError):
        m.on_train_batch_end(outputs=["not", "a", "mapping"], batch=batch, batch_idx=0)  # type: ignore[arg-type]


# -----------------------------------------------------------------------------
# base_embedder.py (deprecated stats helper)
# -----------------------------------------------------------------------------


class _StatsOnlyEmbedder(BaseEmbedderDEPRECATED):
    # Stubs, not used in these tests.
    def preprocess_dataset(self, data):
        raise NotImplementedError

    def compute_pairwise_distances(self, datasets, symmetric=True):
        raise NotImplementedError

    def embed_distance_matrix(self, distance_matrix, emb_dim=None):
        raise NotImplementedError

    def augment_features(self, data, label_embeddings, dataset_idx, class_offsets):
        raise NotImplementedError


def test_get_class_statistics_single_sample_covariance_zero():
    e = _StatsOnlyEmbedder(emb_dim=2, device="cpu")
    X = torch.tensor([[1.0, 2.0], [10.0, 20.0], [3.0, 4.0]])
    Y = torch.tensor([0, 1, 0])

    means, covs = e.get_class_statistics(X, Y)
    assert means.shape == (2, 2)
    assert covs.shape == (2, 2, 2)

    # Label 1 has a single sample -> covariance must be zeros.
    # torch.unique sorts labels; label=0 is idx0, label=1 is idx1.
    assert torch.allclose(covs[1], torch.zeros(2, 2))

