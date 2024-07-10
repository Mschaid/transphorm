import os
from pathlib import Path

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
from comet_ml import Experiment
from dotenv import load_dotenv
from pytorch_lightning.loggers import CometLogger
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from transphorm.model_components.data_objects import SyntheticFPDataModule
from transphorm.model_components.model_modules import (
    AutoEncoder,
    CNNDecoder,
    CNNEncoder,
)


def init_comet_logger():
    load_dotenv()
    COMET_API_KEY = os.getenv("COMET_API_KEY")
    logger = CometLogger(
        api_key=COMET_API_KEY,
        workspace="transphornm",
        project_name="autoencoders",
        experiment_name="cnn_autoencoder",
    )
    return logger


def load_data_module(data_path: Path, batch_size: int = 32) -> L.LightningModule:
    """takes data path and loads da"""
    datamod = SyntheticFPDataModule(batch_size=batch_size, data_path=data_path)
    datamod.prepare_data()
    datamod.setup("fit")
    return datamod


def build_model() -> AutoEncoder:
    ae = AutoEncoder(encoder=CNNEncoder, decoder=CNNDecoder, optimizer=torch.optim.Adam)
    return ae


def train_model(model, datamodule, logger, epochs=1000):
    trainer = L.Trainer(
        accelerator="gpu", devices="auto", max_epochs=epochs, logger=logger
    )
    trainer.fit(model, datamodule)


def main():
    DATA_PATH = "/home/mds8301/data/synthetic_data/synthetic_dataset.pt"

    logger = init_comet_logger()
    datamod = load_data_module(DATA_PATH)
    model = build_model()
    logger.log_hyperparams(model.hparams)
    train_model(model, datamod, logger)


if __name__ == "__main__":
    main()
