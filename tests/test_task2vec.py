import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset

from data_meta_map.task2vec.task2vec import (
    Embedding,
    ProbeNetwork,
    Task2Vec,
    task2vec,
)
from data_meta_map.task2vec.task_similarity import (
    get_hessian,
    get_variance,
    kl,
    jsd,
    cosine,
    correlation,
    pdist,
    cdist,
)
from data_meta_map.task2vec.utils import AverageMeter, get_error, get_device
from data_meta_map.models import get_model
from data_meta_map import datasets


# ── helpers ────────────────────────────────────────────────────────────────────

def _make_embedding(n=8, seed=0):
    rng = np.random.default_rng(seed)
    hess = np.abs(rng.standard_normal(n)) + 0.1
    scale = np.ones(n)
    return Embedding(hessian=hess, scale=scale)


class _SimpleProbeNetwork(ProbeNetwork):
    """Minimal ProbeNetwork for unit-testing Task2Vec.__init__."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.fc = nn.Linear(16, num_classes)
        self.layers = [self.fc]

    @property
    def classifier(self):
        return self.fc

    @classifier.setter
    def classifier(self, val):
        self.fc = val

    def forward(self, x, start_from=0):
        return self.fc(x)


# ── Embedding ──────────────────────────────────────────────────────────────────

class TestEmbedding:
    def test_stores_as_ndarray(self):
        hess = [1.0, 2.0, 3.0]
        scale = [1.0, 1.0, 1.0]
        e = Embedding(hessian=hess, scale=scale)
        assert isinstance(e.hessian, np.ndarray)
        assert isinstance(e.scale, np.ndarray)
        np.testing.assert_array_equal(e.hessian, hess)

    def test_meta_default_none(self):
        e = Embedding(hessian=[1.0], scale=[1.0])
        assert e.meta is None

    def test_meta_stored(self):
        e = Embedding(hessian=[1.0], scale=[1.0], meta={"task": "test"})
        assert e.meta == {"task": "test"}

    def test_tensor_input_converted(self):
        hess = torch.tensor([1.0, 2.0])
        e = Embedding(hessian=hess.numpy(), scale=np.ones(2))
        assert isinstance(e.hessian, np.ndarray)


# ── ProbeNetwork ───────────────────────────────────────────────────────────────

class TestProbeNetwork:
    def test_abstract_cannot_instantiate(self):
        with pytest.raises(TypeError):
            ProbeNetwork()

    def test_concrete_instantiates(self):
        net = _SimpleProbeNetwork()
        assert isinstance(net, ProbeNetwork)

    def test_classifier_property(self):
        net = _SimpleProbeNetwork()
        assert net.classifier is net.fc

# ── Task2Vec.__init__ ──────────────────────────────────────────────────────────


class TestTask2VecInit:
    def test_default_attributes(self):
        model = _SimpleProbeNetwork()
        t2v = Task2Vec(model)
        assert t2v.model is model
        assert t2v.skip_layers == 0
        assert t2v.max_samples is None
        assert t2v.method == "montecarlo"
        assert t2v.bernoulli is False

    def test_custom_attributes(self):
        model = _SimpleProbeNetwork()
        t2v = Task2Vec(model, skip_layers=1, max_samples=100,
                       method="variational", bernoulli=True)
        assert t2v.skip_layers == 1
        assert t2v.max_samples == 100
        assert t2v.method == "variational"
        assert t2v.bernoulli is True

    def test_invalid_method_raises(self):
        model = _SimpleProbeNetwork()
        with pytest.raises(AssertionError):
            Task2Vec(model, method="invalid")

    def test_negative_skip_layers_raises(self):
        model = _SimpleProbeNetwork()
        with pytest.raises(AssertionError):
            Task2Vec(model, skip_layers=-1)

    def test_device_set_from_model(self):
        model = _SimpleProbeNetwork()
        t2v = Task2Vec(model)
        assert t2v.device == torch.device("cpu")

    def test_default_dicts_initialized(self):
        model = _SimpleProbeNetwork()
        t2v = Task2Vec(model)
        assert isinstance(t2v.classifier_opts, dict)
        assert isinstance(t2v.method_opts, dict)
        assert isinstance(t2v.loader_opts, dict)

    def test_loss_fn_cross_entropy_by_default(self):
        model = _SimpleProbeNetwork()
        t2v = Task2Vec(model)
        assert isinstance(t2v.loss_fn, nn.CrossEntropyLoss)

    def test_loss_fn_bce_when_bernoulli(self):
        model = _SimpleProbeNetwork()
        t2v = Task2Vec(model, bernoulli=True)
        assert isinstance(t2v.loss_fn, nn.BCEWithLogitsLoss)

    def test_inherits_base_embedder(self):
        from data_meta_map.base_embedder import BaseEmbedder
        model = _SimpleProbeNetwork()
        t2v = Task2Vec(model)
        assert isinstance(t2v, BaseEmbedder)


# ── Task2Vec.extract_embedding ─────────────────────────────────────────────────

class TestExtractEmbeddingRealData:
    def _make_model_with_grad2(self, n_filters=4):
        model = _SimpleProbeNetwork(num_classes=2)
        # Simulate what montecarlo_fisher stores on weight tensors
        for name, module in model.named_modules():
            if module is model.classifier:
                continue
            if hasattr(module, "weight"):
                module.weight.grad2_acc = torch.ones_like(module.weight) * 0.5
        return model

    def test_mnist_resnet(self):
        dataset = datasets.__dict__['mnist'](root='../../data')[0]
        model = get_model('resnet18', pretrained=True,
                          num_classes=int(max(dataset.targets)+1)).cuda()
        task2vec_embedder = Task2Vec(model, skip_layers=6, max_samples=200)
        emb = task2vec_embedder.embed(dataset)
        assert isinstance(emb, np.ndarray)
        assert emb.shape == (7680, )

    def test_mnist_resnet_less_skip(self):
        dataset = datasets.__dict__['mnist'](root='../../data')[0]
        model = get_model('resnet18', pretrained=True,
                          num_classes=int(max(dataset.targets)+1)).cuda()
        task2vec_embedder = Task2Vec(model, skip_layers=2, max_samples=200)
        emb = task2vec_embedder.embed(dataset)
        assert isinstance(emb, np.ndarray)
        assert emb.shape == (9472,)

    def test_extract_hessian(self):
        dataset = datasets.__dict__['mnist'](root='../../data')[0]
        model = get_model('resnet18', pretrained=True,
                          num_classes=int(max(dataset.targets)+1)).cuda()
        task2vec_embedder = Task2Vec(model, skip_layers=2, max_samples=200)
        emb = task2vec_embedder.embed(dataset, create_final_embedding=False)
        assert isinstance(emb.hessian, np.ndarray)
        assert isinstance(emb.scale, np.ndarray)
        assert emb.scale.shape == (9472,)
        assert emb.hessian.shape == (9472,)


class TestDistanceFunctions:
    @pytest.fixture
    def pair(self):
        return _make_embedding(8, 0), _make_embedding(8, 1)

    def test_get_variance_returns_array(self, pair):
        e = pair[0]
        var = get_variance(e)
        assert isinstance(var, np.ndarray)
        assert var.shape == e.hessian.shape

    def test_get_variance_normalized(self, pair):
        e = pair[0]
        var = get_variance(e, normalized=True)
        assert isinstance(var, np.ndarray)

    def test_get_hessian_returns_array(self, pair):
        e = pair[0]
        h = get_hessian(e)
        assert isinstance(h, np.ndarray)
        np.testing.assert_array_equal(h, e.hessian)

    def test_get_hessian_normalized(self, pair):
        e = pair[0]
        h = get_hessian(e, normalized=True)
        assert isinstance(h, np.ndarray)

    def test_kl_self_is_zero(self):
        e = _make_embedding(8, 0)
        assert kl(e, e) == pytest.approx(0.0, abs=1e-6)

    def test_kl_non_negative(self, pair):
        assert kl(*pair) >= 0.0

    def test_kl_symmetric(self, pair):
        e0, e1 = pair
        assert kl(e0, e1) == pytest.approx(kl(e1, e0), rel=1e-6)

    def test_jsd_self_is_zero(self):
        e = _make_embedding(8, 0)
        assert jsd(e, e) == pytest.approx(0.0, abs=1e-6)

    def test_jsd_non_negative(self, pair):
        assert jsd(*pair) >= 0.0

    def test_cosine_self_is_zero(self):
        e = _make_embedding(8, 0)
        assert cosine(e, e) == pytest.approx(0.0, abs=1e-6)

    def test_cosine_bounded(self, pair):
        d = cosine(*pair)
        assert 0.0 <= d <= 2.0

    def test_correlation_self_is_zero(self):
        e = _make_embedding(8, 0)
        assert correlation(e, e) == pytest.approx(0.0, abs=1e-6)


# ── task_similarity: pdist / cdist ────────────────────────────────────────────

class TestPdistCdist:
    def test_pdist_shape(self):
        embeddings = [_make_embedding(8, i) for i in range(4)]
        D = pdist(embeddings, distance="cosine")
        assert D.shape == (4, 4)

    def test_pdist_diagonal_zero(self):
        embeddings = [_make_embedding(8, i) for i in range(3)]
        D = pdist(embeddings, distance="cosine")
        np.testing.assert_allclose(np.diag(D), 0.0, atol=1e-6)

    def test_pdist_symmetric(self):
        embeddings = [_make_embedding(8, i) for i in range(3)]
        D = pdist(embeddings, distance="cosine")
        np.testing.assert_allclose(D, D.T, atol=1e-6)

    def test_cdist_shape(self):
        src = [_make_embedding(8, i) for i in range(3)]
        tgt = [_make_embedding(8, i + 10) for i in range(2)]
        D = cdist(src, tgt, distance="cosine")
        assert D.shape == (3, 2)

    def test_pdist_kl(self):
        embeddings = [_make_embedding(8, i) for i in range(3)]
        D = pdist(embeddings, distance="kl")
        assert D.shape == (3, 3)
        np.testing.assert_allclose(np.diag(D), 0.0, atol=1e-6)


# ── utils ──────────────────────────────────────────────────────────────────────

class TestAverageMeter:
    def test_initial_state(self):
        m = AverageMeter()
        assert m.avg["loss"] == 0.0

    def test_single_update(self):
        m = AverageMeter()
        m.update(n=4, loss=2.0)
        assert m.avg["loss"] == pytest.approx(2.0)

    def test_multiple_updates_weighted(self):
        m = AverageMeter()
        m.update(n=2, loss=1.0)
        m.update(n=2, loss=3.0)
        assert m.avg["loss"] == pytest.approx(2.0)

    def test_reset_clears_state(self):
        m = AverageMeter()
        m.update(n=1, loss=5.0)
        m.reset()
        assert m.sum["loss"] == 0
        assert m.count["loss"] == 0

    def test_multiple_metrics(self):
        m = AverageMeter()
        m.update(n=1, loss=1.0, error=0.5)
        assert m.avg["loss"] == pytest.approx(1.0)
        assert m.avg["error"] == pytest.approx(0.5)


class TestGetError:
    def test_all_correct(self):
        output = torch.tensor([[0.0, 10.0], [10.0, 0.0]])
        target = torch.tensor([1, 0])
        assert get_error(output, target) == pytest.approx(0.0)

    def test_all_wrong(self):
        output = torch.tensor([[10.0, 0.0], [0.0, 10.0]])
        target = torch.tensor([1, 0])
        assert get_error(output, target) == pytest.approx(100.0)

    def test_half_correct(self):
        output = torch.tensor([[10.0, 0.0], [10.0, 0.0]])
        target = torch.tensor([0, 1])
        assert get_error(output, target) == pytest.approx(50.0)


class TestGetDevice:
    def test_returns_cpu_device(self):
        model = nn.Linear(4, 2)
        device = get_device(model)
        assert device == torch.device("cpu")
