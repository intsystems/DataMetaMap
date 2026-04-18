import pytest
import torch

from data_meta_map.base_embedder import BaseEmbedder


class _ConcreteEmbedder(BaseEmbedder):
    def embed(self, X, y=None):
        return X.mean(dim=0)


def test_abstract_cannot_be_instantiated():
    with pytest.raises(TypeError):
        BaseEmbedder()


def test_concrete_subclass_instantiates():
    e = _ConcreteEmbedder()
    assert isinstance(e, BaseEmbedder)


def test_missing_embed_raises():
    class _Incomplete(BaseEmbedder):
        pass

    with pytest.raises(TypeError):
        _Incomplete()


def test_embed_callable():
    e = _ConcreteEmbedder()
    X = torch.randn(10, 4)
    result = e.embed(X)
    assert result.shape == (4,)
