import os
from pathlib import Path

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from comet_ml import Experiment
import torch
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

from dataclasses import dataclass


@dataclass
class ModelConfig:
    encoder: str = "cnn"
    decoder: str = "cnn"
    optimizer: str = "adam"
    epochs: int = 1000
    learning_rate: float = 1e-4
    batch_size: int = 32


def init_comet_logger():
    load_dotenv()
    COMET_API_KEY = os.getenv("COMET_API_KEY")
    logger = CometLogger(
        api_key=COMET_API_KEY,
        workspace="transphorm",
        project_name="autoencoders",
        experiment_name="synthetic_cnn_autoencoder_v2",
    )
    return logger


def load_data_module(data_path: Path, batch_size: int = 32) -> L.LightningModule:
    """takes data path and loads da"""
    datamod = SyntheticFPDataModule(batch_size=batch_size, data_path=data_path)
    datamod.prepare_data()
    datamod.setup("fit")
    return datamod


def build_model(model_config: ModelConfig) -> AutoEncoder:
    ae = AutoEncoder(
        encoder=CNNEncoder,
        decoder=CNNDecoder,
        optimizer=torch.optim.Adam,
        learning_rate=model_config.learning_rate,
    )
    return ae


def train_model(model, datamodule, logger, epochs=1000):
    trainer = L.Trainer(
        accelerator="gpu", devices="auto", max_epochs=epochs, logger=logger
    )
    trainer.fit(model, datamodule)


def main():
    DATA_PATH = "/Users/mds8301/Desktop/temp/synthetic_dataset.pt"
    model_config = ModelConfig()

    logger = init_comet_logger()
    logger.log_hyperparams(model_config.__dict__)
    datamod = load_data_module(DATA_PATH)
    model = build_model(model_config)
    logger.log_hyperparams(model.hparams)
    train_model(model, datamod, logger)


if __name__ == "__main__":
    main()
