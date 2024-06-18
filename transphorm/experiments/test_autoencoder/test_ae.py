import torch
from torch import nn
import lightning as L
import matplotlib.pyplot as plt
from transphorm.model_components import SyntheticTSDataModule, AADataModule
from pathlib import Path
import seaborn as sns

from transphorm.model_components.model_modules import (
    Encoder,
    CNNENcoder,
    Decoder,
    VanillaAutoEncoder,
)


def main():
    data_path = Path("/Users/mds8301/Desktop/temp/dopamine_full_timeseries_array.pt")
    data_module = AADataModule(main_path=data_path, batch_size=16, num_workers=4)
    data_module.prepare_data()
    data_module.setup("train")
    seq_length = data_module.tensor_shape[-1]
    van_ae = VanillaAutoEncoder(
        encoder=Encoder,
        decoder=Decoder,
        seq_length=seq_length,
        smallest_layer=64,
        optimizer=torch.optim.Adam,
    )
    trainer = L.Trainer(max_epochs=200)
    trainer.fit(van_ae, data_module)


if __name__ == "__main__":
    main()
