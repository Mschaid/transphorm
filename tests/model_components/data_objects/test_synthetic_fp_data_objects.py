import pytest
from transphorm.model_components.data_objects import SyntheticFPDataModule
from torch.utils.data import DataLoader, TensorDataset, Subset


@pytest.fixture
def data_module():
    return SyntheticFPDataModule(batch_size=32, num_workers=1, sample_size=1000)


class TestSyntheticFPDataModule:
    def test_create_sinewave(self, data_module):
        freq = 5
        result = data_module.create_sinewave(freq)
        assert result.shape == (2, 1000)

    def test_create_square_wave(self, data_module):
        freq = 5
        result = data_module.create_square_wave(freq)
        assert result.shape == (2, 1000)

    def test_create_dataset(self, data_module):
        X, y = data_module.create_dataset()
        assert X.shape == (1000, 2, 1000)
        assert y.shape == (1000,)

    def test_prepare_data(self, data_module):
        data_module.prepare_data()
        assert isinstance(data_module.data, TensorDataset)

    def test_setup(self, data_module):
        data_module.prepare_data()
        data_module.setup("train")
        assert isinstance(data_module.train, Subset)
        assert isinstance(data_module.val, Subset)
        assert isinstance(data_module.test, Subset)

    def test_tensor_shape(self, data_module):
        data_module.prepare_data()
        data_module.setup("train")
        assert isinstance(data_module.tensor_shape, tuple)
