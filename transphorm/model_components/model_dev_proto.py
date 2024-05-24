import logging
import os
from pathlib import Path
from typing import Any, Tuple

from comet_ml import Experiment
import lightning as L
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from lightning.pytorch.demos import Transformer
from pytorch_lightning.loggers import CometLogger
from torch import nn
from torch.profiler import ProfilerActivity, profile, record_function
from torch.utils.data import DataLoader, Dataset, TensorDataset

from transphorm.model_components import (
    CNN,
    AADataModule,
    DeepBinaryClassifier,
    DeepLinearLayer,
)

load_dotenv()


def set_up_comet_logger():
    COMET_API_KEY = os.getenv("COMET_API_KEY")
    logger = CometLogger(
        api_key=COMET_API_KEY,
        workspace="mikes-test-ws",
        project_name="test_run_linear_classifer",
        experiment_name="exp_1",
    )

    return logger


def main():
    MAIN_DATA_PATH = Path(
        "/Users/mds8301/Desktop/temp/dopamine_full_timeseries_array.pt"
    )
    BATCH_SIZE = 32
    loader = AADataModule(main_path=MAIN_DATA_PATH, batch_size=BATCH_SIZE)
    # loader = TestTSDataModule(batch_size=batch_size)
    loader.prepare_data()
    loader.setup(stage="train")
    logger = set_up_comet_logger()
    optimzer = torch.optim.Adam

    classifier = DeepLinearLayer(loader.input_size)
    # cnn = CNN()
    optimzer = torch.optim.Adam

    model = DeepBinaryClassifier(classifier, optimizer=optimzer)

    logger.log_hyperparams(model.hparams)
    trainer = L.Trainer(
        accelerator="mps",
        devices=1,
        max_epochs=500,
        profiler=None,
        enable_progress_bar=False,
        logger=logger,
        enable_checkpointing=False,
    )
    trainer.fit(model, loader)


if __name__ == "__main__":
    main()
