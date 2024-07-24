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
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from transphorm.model_components.data_objects import SyntheticFPDataModule
from transphorm.model_components.model_modules import (
    AutoEncoder,
    BranchingCNNDecoder,
    BranchingCNNEncoder,
)

from dataclasses import dataclass

# incase methods aren't on mps
if torch.backends.mps.is_available():
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


@dataclass
class ModelConfig:
    encoder: str = "cnn"
    decoder: str = "cnn"
    optimizer: str = "adam"
    experiment_name: str = "unnamed"
    epochs: int = 1000
    learning_rate: float = 1e-4
    batch_size: int = 32


def init_comet_logger(experiment_name: str):
    load_dotenv()
    COMET_API_KEY = os.getenv("COMET_API_KEY")
    logger = CometLogger(
        api_key=COMET_API_KEY,
        workspace="transphorm",
        project_name="autoencoders",
        experiment_name=experiment_name,
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
        encoder=BranchingCNNEncoder(),
        decoder=BranchingCNNDecoder(),
        optimizer=torch.optim.Adam,
        learning_rate=model_config.learning_rate,
    )
    return ae


def train_model(
    model: L.LightningModule,
    datamodule: L.LightningDataModule,
    logger: CometLogger,
    root_dir: Path,
    model_configs: ModelConfig,
):
    checkpint_callback = ModelCheckpoint(
        filename=model_configs.experiment_name,
    )
    trainer = L.Trainer(
        accelerator="gpu",
        devices="auto",
        max_epochs=model_configs.epochs,
        logger=logger,
        default_root_dir=root_dir,
        callbacks=[checkpint_callback],
    )
    trainer.fit(model, datamodule)


def main():
    DATA_PATH = "/Users/mds8301/Desktop/temp/synthetic_dataset.pt"
    model_config = ModelConfig(
        encoder="branching_dilated_cnn_w_attention",
        decoder="branching_dilated_cnn_w_attention",
        experiment_name="branching_dilated_cnn_w_attention_v1",
    )

    LOG_DIR = Path("/Users/mds8301/Development/transphorm/models/experiments")

    logger = init_comet_logger(model_config.experiment_name)
    logger.log_hyperparams(model_config.__dict__)
    datamod = load_data_module(DATA_PATH)
    model = build_model(model_config)
    logger.log_hyperparams(model.hparams)
    train_model(model, datamod, logger, root_dir=LOG_DIR, model_configs=model_config)


if __name__ == "__main__":
    main()
