import numpy as np
import pytest
import torch

from data_meta_map.dataset2vec.config import Dataset2VecConfig, OptimizerConfig
from data_meta_map.dataset2vec.model import Dataset2Vec
from data_meta_map.dataset2vec_embedder import Dataset2VecEmbedder, dataset2vec


@pytest.fixture
def model():
    return Dataset2Vec(Dataset2VecConfig(), OptimizerConfig())


@pytest.fixture
def embedder(model):
    return Dataset2VecEmbedder(model)


@pytest.fixture
def fitted_embedder(embedder):
    embedder._is_fitted = True
    return embedder


# ── init ──────────────────────────────────────────────────────────────────────

def test_init_attributes(model):
    e = Dataset2VecEmbedder(model)
    assert e.model is model
    assert e.max_epochs == 10
    assert e.batch_size == 32
    assert e.n_batches == 100
    assert e._is_fitted is False


def test_init_custom_params(model):
    e = Dataset2VecEmbedder(model, max_epochs=5, batch_size=16, n_batches=50)
    assert e.max_epochs == 5
    assert e.batch_size == 16
    assert e.n_batches == 50


# ── embed ─────────────────────────────────────────────────────────────────────

def test_embed_before_fit_raises(embedder):
    X = torch.randn(10, 5)
    y = torch.randint(0, 2, (10,)).float()
    with pytest.raises(RuntimeError):
        embedder.embed(X, y)


def test_embed_returns_ndarray(fitted_embedder):
    X = torch.randn(10, 5)
    y = torch.randint(0, 2, (10,)).float()
    result = fitted_embedder.embed(X, y)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 1
    assert result.shape[0] == fitted_embedder.model.output_size


def test_embed_output_shape_matches_config(model):
    cfg = Dataset2VecConfig(output_size=8)
    e = Dataset2VecEmbedder(Dataset2Vec(cfg, OptimizerConfig()))
    e._is_fitted = True
    X = torch.randn(10, 3)
    y = torch.randint(0, 2, (10,)).float()
    result = e.embed(X, y)
    assert result.shape == (8,)


# ── save / load ───────────────────────────────────────────────────────────────

def test_save_and_load(fitted_embedder, tmp_path):
    path = str(tmp_path / "weights.pt")
    fitted_embedder.save(path)

    e2 = Dataset2VecEmbedder(Dataset2Vec(Dataset2VecConfig(), OptimizerConfig()))
    assert not e2._is_fitted
    result = e2.load(path)
    assert result is e2
    assert e2._is_fitted


def test_load_returns_self(embedder, tmp_path):
    embedder._is_fitted = True
    path = str(tmp_path / "w.pt")
    embedder.save(path)
    ret = embedder.load(path)
    assert ret is embedder


# ── dataset2vec convenience function ─────────────────────────────────────────

def test_dataset2vec_raises_without_fit(model):
    X = torch.randn(10, 5)
    y = torch.randint(0, 2, (10,)).float()
    with pytest.raises(RuntimeError):
        dataset2vec(model, X, y, fit_data=None)
