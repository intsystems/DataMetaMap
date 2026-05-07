import pytest
import torch
from torch.utils.data import Dataset, DataLoader

from data_meta_map.wasserstein_embedder import WassersteinEmbedder


class MockDataset(Dataset):
    def __init__(self, n=100, d=10, k=5, seed=42):
        torch.manual_seed(seed)
        self.data = torch.randn(n, d)
        self.labels = torch.randint(0, k, (n,))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


@pytest.fixture
def embedder():
    return WassersteinEmbedder(emb_dim=2)


@pytest.fixture
def small_ds():
    return MockDataset(n=50, d=10, k=3, seed=42)


# ── init ──────────────────────────────────────────────────────────────────────

def test_init_defaults(embedder):
    assert embedder.emb_dim == 2
    assert embedder.device == torch.device("cpu")
    assert embedder.max_samples is None
    assert embedder.batch_size == 64
    assert embedder.gaussian_assumption is True
    assert embedder.diagonal_cov is False


def test_init_custom_params():
    e = WassersteinEmbedder(
        emb_dim=5, device="cpu", max_samples=50, batch_size=32,
        gaussian_assumption=False, diagonal_cov=True, sqrt_niters=10,
    )
    assert e.emb_dim == 5
    assert e.max_samples == 50
    assert e.batch_size == 32
    assert e.gaussian_assumption is False
    assert e.diagonal_cov is True
    assert e.sqrt_niters == 10


def test_default_emb_dim():
    e = WassersteinEmbedder()
    assert e.emb_dim == 2


# ── preprocess_dataset ────────────────────────────────────────────────────────

def test_preprocess_dataset_shape(embedder, small_ds):
    X, Y = embedder.preprocess_dataset(small_ds, dataset_id=0)
    assert X.shape == (50, 10)
    assert Y.shape == (50,)
    assert X.dtype == torch.float32
    assert Y.dtype == torch.long


def test_preprocess_via_dataloader(embedder, small_ds):
    loader = DataLoader(small_ds, batch_size=16)
    X, Y = embedder.preprocess_dataset(loader, dataset_id=1)
    assert X.shape == (50, 10)
    assert Y.shape == (50,)


def test_preprocess_max_samples():
    e = WassersteinEmbedder(emb_dim=2, max_samples=30)
    ds = MockDataset(n=100, d=10, k=5)
    X, Y = e.preprocess_dataset(ds, dataset_id=0)
    assert X.shape[0] == 30


def test_preprocess_caching(embedder, small_ds):
    X1, Y1 = embedder.preprocess_dataset(small_ds, dataset_id=0)
    X2, Y2 = embedder.preprocess_dataset(small_ds, dataset_id=0)
    assert torch.equal(X1, X2)
    assert torch.equal(Y1, Y2)


def test_preprocess_invalid_type(embedder):
    with pytest.raises(TypeError):
        embedder.preprocess_dataset("not_a_dataset")


# ── _compute_gaussian_stats ───────────────────────────────────────────────────

def test_gaussian_stats_full(embedder):
    X = torch.randn(100, 10)
    Y = torch.tensor([0] * 50 + [1] * 50)
    means, covs, offsets = embedder._compute_gaussian_stats(X, Y)
    assert means.shape == (2, 10)
    assert covs.shape == (2, 10, 10)
    assert offsets == [0, 1]


def test_gaussian_stats_diagonal():
    e = WassersteinEmbedder(emb_dim=2, diagonal_cov=True)
    X = torch.randn(100, 10)
    Y = torch.tensor([0] * 30 + [1] * 30 + [2] * 40)
    means, covs, offsets = e._compute_gaussian_stats(X, Y)
    assert means.shape == (3, 10)
    assert covs.shape == (3, 10)  # diagonal stored as vector


# ── _bures_wasserstein_distance ───────────────────────────────────────────────

def test_bures_distance_identical_dists(embedder):
    mean = torch.zeros(2)
    cov = torch.eye(2)
    d = embedder._bures_wasserstein_distance(mean, cov, mean, cov)
    assert torch.allclose(d, torch.tensor(0.0), atol=1e-2)


def test_bures_distance_different_means(embedder):
    m1 = torch.tensor([0.0, 0.0])
    m2 = torch.tensor([1.0, 0.0])
    cov = torch.eye(2)
    d = embedder._bures_wasserstein_distance(m1, cov, m2, cov)
    assert d > 0.0
    assert torch.allclose(d, torch.tensor(1.0), atol=1e-5)


def test_bures_distance_symmetry(embedder):
    torch.manual_seed(0)
    m1, m2 = torch.randn(4), torch.randn(4)
    A = torch.randn(4, 4)
    cov = A @ A.T + torch.eye(4) * 0.1
    d12 = embedder._bures_wasserstein_distance(m1, cov, m2, cov)
    d21 = embedder._bures_wasserstein_distance(m2, cov, m1, cov)
    assert torch.allclose(d12, d21, atol=1e-4)


# ── compute_pairwise_distances ────────────────────────────────────────────────

def test_pairwise_distances_single_dataset():
    e = WassersteinEmbedder(emb_dim=2, max_samples=30)
    ds = MockDataset(n=50, d=10, k=3)
    D = e.compute_pairwise_distances([ds])
    assert D.shape == (3, 3)
    assert torch.all(D >= 0)
   # assert torch.allclose(D, D.T, atol=1e-5)
  #  assert torch.allclose(D.diag(), torch.zeros(3), atol=1e-2)


def test_pairwise_distances_multiple_datasets():
    e = WassersteinEmbedder(emb_dim=2, max_samples=20)
    ds1 = MockDataset(n=30, d=10, k=2, seed=1)
    ds2 = MockDataset(n=30, d=10, k=3, seed=2)
    D = e.compute_pairwise_distances([ds1, ds2])
    assert D.shape == (5, 5)
    assert torch.all(D >= 0)


def test_pairwise_distances_diagonal_mode():
    e = WassersteinEmbedder(emb_dim=2, max_samples=20, diagonal_cov=True)
    ds = MockDataset(n=40, d=8, k=3)
    D = e.compute_pairwise_distances([ds])
    assert D.shape == (3, 3)
    assert torch.all(D >= 0)


# ── embed_distance_matrix ─────────────────────────────────────────────────────

def test_embed_distance_matrix_shape(embedder):
    points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    D = torch.cdist(points, points)
    embs = embedder.embed_distance_matrix(D, emb_dim=2)
    assert embs.shape == (4, 2)


def test_embed_distance_matrix_uses_default_dim():
    e = WassersteinEmbedder(emb_dim=3)
    D = torch.zeros(5, 5)
    embs = e.embed_distance_matrix(D)
    assert embs.shape == (5, 3)


# ── augment_features ──────────────────────────────────────────────────────────

def test_augment_features_shape(embedder, small_ds):
    label_embs = torch.randn(3, 2)
    Z = embedder.augment_features(small_ds, label_embs, 0, [0, 3])
    assert Z.shape == (50, 12)  # 10 features + 2 label emb dims


# ── clear_cache ───────────────────────────────────────────────────────────────

def test_clear_cache(embedder, small_ds):
    embedder.preprocess_dataset(small_ds, dataset_id=0)
    assert len(embedder._data_cache) > 0
    embedder.clear_cache()
    assert len(embedder._data_cache) == 0
    assert len(embedder._stats_cache) == 0


# ── embed (BaseEmbedder interface) ────────────────────────────────────────────

def test_embed_method_returns_wte():
    e = WassersteinEmbedder(emb_dim=2, max_samples=20)
    ds = MockDataset(n=20, d=8, k=2)
    task_embs, label_embs, aug_data = e.embed([ds])
    assert task_embs.shape[0] == 1
    assert label_embs.shape[1] == 2
    assert len(aug_data) == 1
