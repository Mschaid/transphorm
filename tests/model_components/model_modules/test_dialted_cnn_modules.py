import pytest
import torch
import lightning as L
from transphorm.model_components.model_modules import (
    DilatedCNNEncoder,
    DilatedCNNDecoder,
    BigDilatedCNNDecoder,
    BigDilatedCNNEncoder,
    XLDilatedCNNEncoder,
    XLDilatedCNNDecoder,
)
from transphorm.model_components.model_modules import AutoEncoder


@pytest.fixture
def data():
    return torch.rand(3, 1000)


@pytest.fixture
def latents():
    return torch.rand(128)


@pytest.fixture
def big_latents():
    return torch.rand(512)


@pytest.fixture
def xl_latents():
    return torch.rand(1024)


class TestDilatedCNNEncoder:

    def test_forward(self, data):
        encoder = DilatedCNNEncoder()
        output = encoder(data)
        assert output.shape == torch.Size([128])


class TestDilatedCNNDecoder:

    def test_forward(self, latents):
        decoder = DilatedCNNDecoder()
        output = decoder(latents)
        assert output.shape == torch.Size([3, 1000])


class TestBigDilatedCNNEncoder:

    def test_forward(self, data):
        encoder = BigDilatedCNNEncoder()
        output = encoder(data)
        assert output.shape == torch.Size([512])


class TestBigDilatedCNNDecoder:

    def test_forward(self, big_latents):
        decoder = BigDilatedCNNDecoder()
        output = decoder(big_latents)
        assert output.shape == torch.Size([3, 1000])


class TestXLDilatedCNNEncoder:

    def test_forward(self, data):
        encoder = XLDilatedCNNEncoder()
        output = encoder(data)
        assert output.shape == torch.Size([1024])


class TestXLDilatedCNNDecoder:
    def test_forward(self, xl_latents):
        decoder = XLDilatedCNNDecoder()
        output = decoder(xl_latents)
        assert output.shape == torch.Size([3, 1000])
