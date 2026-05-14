import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from data_meta_map.mmd import (
    IMQKernel,
    LinearKernel,
    PolynomialKernel,
    RBFKernel,
    RFFKernelMean,
    gram_matrix,
    get_kernel,
    median_heuristic,
    mmd2_biased,
    mmd2_linear,
    mmd2_unbiased,
)
from data_meta_map.mmd.encoders import (
    GenerativeEncoder,
    PretrainedEncoder,
    RawEncoder,
)
from data_meta_map.mmd_embedder import MMDEmbedder, mmd


class MockDataset(Dataset):
    def __init__(self, n=100, d=10, k=5, shift=0.0, seed=42):
        torch.manual_seed(seed)
        self.data = torch.randn(n, d) + shift
        self.labels = torch.randint(0, k, (n,))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# ── kernels ───────────────────────────────────────────────────────────────────

def test_rbf_kernel_self_is_one():
    k = RBFKernel(sigma=1.0)
    X = torch.randn(8, 3)
    K = k(X, X)
    assert torch.allclose(torch.diagonal(K), torch.ones(8), atol=1e-6)


def test_rbf_kernel_symmetric():
    k = RBFKernel(sigma=2.0)
    X = torch.randn(5, 4)
    Y = torch.randn(7, 4)
    K_xy = k(X, Y)
    K_yx = k(Y, X)
    assert torch.allclose(K_xy, K_yx.T, atol=1e-6)


def test_rbf_kernel_matches_naive_loop():
    torch.manual_seed(0)
    sigma = 1.5
    k = RBFKernel(sigma=sigma)
    X = torch.randn(6, 3)
    Y = torch.randn(4, 3)
    K_batched = k(X, Y)
    K_naive = torch.zeros(6, 4)
    for i in range(6):
        for j in range(4):
            K_naive[i, j] = torch.exp(
                -((X[i] - Y[j]) ** 2).sum() / (2 * sigma ** 2)
            )
    assert torch.allclose(K_batched, K_naive, atol=1e-6)


def test_rbf_auto_bandwidth_via_median():
    k = RBFKernel(sigma=None)
    X = torch.randn(20, 5)
    K = k(X, X)
    assert k.sigma is not None and k.sigma > 0
    assert K.shape == (20, 20)


def test_linear_polynomial_imq_kernels_shape():
    X = torch.randn(5, 3)
    Y = torch.randn(7, 3)
    assert LinearKernel()(X, Y).shape == (5, 7)
    assert PolynomialKernel(degree=2)(X, Y).shape == (5, 7)
    assert IMQKernel(c=1.0, beta=0.5)(X, Y).shape == (5, 7)


def test_polynomial_invalid_degree_raises():
    with pytest.raises(ValueError):
        PolynomialKernel(degree=0)


def test_imq_invalid_beta_raises():
    with pytest.raises(ValueError):
        IMQKernel(c=1.0, beta=0.0)


def test_get_kernel_string_and_passthrough():
    rbf = get_kernel("rbf", sigma=1.0)
    assert isinstance(rbf, RBFKernel)
    assert get_kernel(rbf) is rbf


def test_get_kernel_unknown_name_raises():
    with pytest.raises(ValueError):
        get_kernel("nope")


def test_median_heuristic_positive_scalar():
    torch.manual_seed(0)
    X = torch.randn(50, 8)
    sigma = median_heuristic(X)
    assert sigma.numel() == 1 and float(sigma) > 0


def test_median_heuristic_singleton_returns_default():
    X = torch.zeros(1, 4)
    sigma = median_heuristic(X)
    assert float(sigma) > 0


def test_gram_matrix_shape():
    X = torch.randn(5, 3)
    Y = torch.randn(8, 3)
    K = gram_matrix(X, Y, kernel="rbf", sigma=1.0)
    assert K.shape == (5, 8)


# ── MMD^2 estimators ──────────────────────────────────────────────────────────

def test_mmd2_biased_zero_for_same_distribution():
    torch.manual_seed(0)
    X = torch.randn(500, 5)
    Y = torch.randn(500, 5)
    val = float(mmd2_biased(X, Y, kernel="rbf", sigma=1.0))
    assert abs(val) < 0.05


def test_mmd2_unbiased_near_zero_for_same_distribution():
    torch.manual_seed(0)
    X = torch.randn(500, 5)
    Y = torch.randn(500, 5)
    val = float(mmd2_unbiased(X, Y, kernel="rbf", sigma=1.0))
    assert abs(val) < 0.05


def test_mmd2_grows_with_mean_shift():
    torch.manual_seed(0)
    X = torch.randn(500, 4)
    Y = torch.randn(500, 4) + 3.0
    val = float(mmd2_biased(X, Y, kernel="rbf", sigma=1.0))
    assert val > 0.1


def test_mmd2_unbiased_requires_min_samples():
    X = torch.randn(1, 3)
    Y = torch.randn(5, 3)
    with pytest.raises(ValueError):
        mmd2_unbiased(X, Y, kernel="rbf", sigma=1.0)


def test_mmd2_linear_runs_and_is_finite():
    torch.manual_seed(0)
    X = torch.randn(100, 4)
    Y = torch.randn(100, 4) + 1.0
    val = float(mmd2_linear(X, Y, kernel="rbf", sigma=1.0))
    assert torch.isfinite(torch.tensor(val))


def test_mmd2_linear_too_few_samples_raises():
    X = torch.randn(1, 3)
    Y = torch.randn(1, 3)
    with pytest.raises(ValueError):
        mmd2_linear(X, Y, kernel="rbf", sigma=1.0)


# ── RFF mean embedding ───────────────────────────────────────────────────────

def test_rff_inner_product_approximates_rbf():
    torch.manual_seed(0)
    sigma = 1.0
    n_rff = 4096
    X = torch.randn(20, 4)
    Y = torch.randn(20, 4) + 0.5
    K_true = RBFKernel(sigma=sigma)(X, Y)
    rff = RFFKernelMean(d=4, n_features=n_rff, sigma=sigma, seed=0)
    zx = rff.features(X)
    zy = rff.features(Y)
    K_approx = zx @ zy.T
    assert (K_true - K_approx).abs().mean() < 0.05


def test_rff_embed_shape():
    rff = RFFKernelMean(d=6, n_features=128, sigma=1.0, seed=0)
    X = torch.randn(40, 6)
    mu = rff.embed(X)
    assert mu.shape == (128,)


def test_rff_fit_bandwidth_sets_sigma():
    rff = RFFKernelMean(d=4, n_features=64, seed=0)
    assert rff.sigma is None
    X = torch.randn(40, 4)
    rff.fit_bandwidth(X)
    assert rff.sigma is not None and rff.sigma > 0


def test_rff_embed_many_stacks():
    rff = RFFKernelMean(d=3, n_features=32, sigma=1.0, seed=0)
    Xs = [torch.randn(20, 3) for _ in range(4)]
    Z = rff.embed_many(Xs)
    assert Z.shape == (4, 32)


def test_rff_invalid_dim_raises():
    with pytest.raises(ValueError):
        RFFKernelMean(d=0, n_features=10)


# ── encoders ─────────────────────────────────────────────────────────────────

def test_raw_encoder_passthrough():
    enc = RawEncoder()
    X = torch.randn(7, 4)
    assert torch.equal(enc.transform(X), X)


def test_generative_encoder_changes_output_dim():
    torch.manual_seed(0)
    enc = GenerativeEncoder(latent_dim=8, hidden_dim=16, epochs=1, seed=0)
    X = torch.randn(40, 6)
    Z = enc.fit(X).transform(X)
    assert Z.shape == (40, 8)


def test_generative_encoder_input_dim_mismatch_raises():
    enc = GenerativeEncoder(latent_dim=4, hidden_dim=8, epochs=1, seed=0)
    enc.fit(torch.randn(20, 5))
    with pytest.raises(ValueError):
        enc.transform(torch.randn(20, 7))


class _TinyImageNet(nn.Module):
    """Tiny image-shaped network with the same children pattern as ResNet."""

    def __init__(self):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.fc = nn.Linear(4, 10)

    def forward(self, x):
        return self.fc(self.feature(x))


def test_pretrained_encoder_with_custom_model():
    enc = PretrainedEncoder(
        model_name="ignored",
        model=_TinyImageNet(),
        image_shape=(3, 8, 8),
        batch_size=4,
        device="cpu",
    )
    X = torch.randn(6, 3 * 8 * 8)
    Z = enc.transform(X)
    assert Z.shape[0] == 6 and Z.shape[1] > 0


def test_pretrained_encoder_invalid_input_shape_raises():
    enc = PretrainedEncoder(
        model_name="ignored",
        model=_TinyImageNet(),
        image_shape=(3, 8, 8),
        device="cpu",
    )
    with pytest.raises(ValueError):
        enc.transform(torch.randn(5, 7))  # not divisible by 3*8*8


# ── MMDEmbedder ──────────────────────────────────────────────────────────────

@pytest.fixture
def small_ds():
    return MockDataset(n=80, d=6, k=3, seed=0)


def test_mmdembedder_init_defaults():
    e = MMDEmbedder()
    assert e.mode == "distance"
    assert e.estimator == "unbiased"
    assert e.emb_dim == 2
    assert e.n_rff == 512
    assert isinstance(e.encoder, RawEncoder)


def test_mmdembedder_invalid_mode_raises():
    with pytest.raises(ValueError):
        MMDEmbedder(mode="nope")


def test_mmdembedder_invalid_estimator_raises():
    with pytest.raises(ValueError):
        MMDEmbedder(estimator="nope")


def test_mmdembedder_preprocess_returns_tensor(small_ds):
    e = MMDEmbedder(max_samples=None, batch_size=16)
    X = e.preprocess_dataset(small_ds, dataset_id=0)
    assert X.shape == (80, 6)
    assert X.dtype == torch.float32


def test_mmdembedder_preprocess_via_dataloader(small_ds):
    e = MMDEmbedder(max_samples=None, batch_size=8)
    loader = DataLoader(small_ds, batch_size=8)
    X = e.preprocess_dataset(loader, dataset_id=1)
    assert X.shape == (80, 6)


def test_mmdembedder_preprocess_caches(small_ds):
    e = MMDEmbedder(max_samples=None, batch_size=16)
    X1 = e.preprocess_dataset(small_ds, dataset_id=0)
    X2 = e.preprocess_dataset(small_ds, dataset_id=0)
    assert X1 is X2  # cache returns identical object


def test_mmdembedder_preprocess_invalid_type_raises():
    e = MMDEmbedder()
    with pytest.raises(TypeError):
        e.preprocess_dataset("not a dataset", dataset_id=0)


def test_mmdembedder_pairwise_matrix_shape_and_symmetry(small_ds):
    e = MMDEmbedder(max_samples=None, batch_size=16, estimator="biased")
    ds_b = MockDataset(n=80, d=6, k=3, shift=1.0, seed=1)
    D = e.compute_pairwise_distances([small_ds, ds_b])
    assert D.shape == (2, 2)
    assert torch.allclose(D, D.T, atol=1e-5)
    assert torch.allclose(D.diag(), torch.zeros(2), atol=1e-6)


def test_mmdembedder_distance_mode_embedding_shape(small_ds):
    e = MMDEmbedder(mode="distance", emb_dim=2, max_samples=None, batch_size=16)
    ds_b = MockDataset(n=80, d=6, k=3, shift=1.0, seed=1)
    ds_c = MockDataset(n=80, d=6, k=3, shift=2.0, seed=2)
    vecs = e.embed([small_ds, ds_b, ds_c])
    assert vecs.shape == (3, 2)


def test_mmdembedder_rff_mode_embedding_shape(small_ds):
    e = MMDEmbedder(mode="rff", n_rff=64, max_samples=None, batch_size=16, seed=0)
    ds_b = MockDataset(n=80, d=6, k=3, shift=1.0, seed=1)
    vecs = e.embed([small_ds, ds_b])
    assert vecs.shape == (2, 64)


def test_mmdembedder_rff_mode_rejects_non_rbf_kernel(small_ds):
    e = MMDEmbedder(mode="rff", kernel="linear", max_samples=None)
    with pytest.raises(ValueError):
        e.embed([small_ds])


def test_mmdembedder_clear_cache(small_ds):
    e = MMDEmbedder(mode="rff", n_rff=32, seed=0)
    e.embed([small_ds])
    assert len(e._feat_cache) > 0
    assert e._rff is not None
    e.clear_cache()
    assert len(e._feat_cache) == 0
    assert e._rff is None
    assert e._kernel is None


def test_mmdembedder_with_generative_encoder_runs(small_ds):
    enc = GenerativeEncoder(latent_dim=4, hidden_dim=8, epochs=1, seed=0)
    e = MMDEmbedder(mode="distance", encoder=enc, emb_dim=2, max_samples=None)
    ds_b = MockDataset(n=60, d=6, k=3, shift=1.0, seed=1)
    vecs = e.embed([small_ds, ds_b])
    assert vecs.shape == (2, 2)


def test_mmdembedder_with_pretrained_encoder_runs():
    enc = PretrainedEncoder(
        model_name="ignored",
        model=_TinyImageNet(),
        image_shape=(3, 8, 8),
        batch_size=4,
        device="cpu",
    )
    e = MMDEmbedder(mode="distance", encoder=enc, emb_dim=2, max_samples=None)

    class ImgDS(Dataset):
        def __init__(self, shift):
            torch.manual_seed(0 if shift == 0.0 else 1)
            self.data = torch.randn(12, 3, 8, 8) + shift
            self.labels = torch.randint(0, 3, (12,))

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx], self.labels[idx]

    vecs = e.embed([ImgDS(0.0), ImgDS(2.0)])
    assert vecs.shape == (2, 2)


def test_mmdembedder_accepts_tensor_inputs():
    e = MMDEmbedder(mode="distance", emb_dim=2, max_samples=None)
    A = torch.randn(40, 5)
    B = torch.randn(40, 5) + 1.0
    vecs = e.embed([A, B])
    assert vecs.shape == (2, 2)


# ── mmd() convenience function ───────────────────────────────────────────────

def test_mmd_convenience_two_tensors():
    torch.manual_seed(0)
    A = torch.randn(60, 4)
    B = torch.randn(60, 4) + 2.0
    val = float(mmd(A, B, kernel="rbf", estimator="biased"))
    assert val > 0


def test_mmd_convenience_zero_for_identical():
    torch.manual_seed(0)
    X = torch.randn(200, 4)
    val = float(mmd(X, X, kernel="rbf", estimator="biased"))
    assert abs(val) < 1e-3
