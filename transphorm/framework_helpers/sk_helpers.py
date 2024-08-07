from pathlib import Path
import torch
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    accuracy_score,
)
from sklearn.model_selection import train_test_split

from transphorm.model_components.data_objects import AATrialDataModule


def load_py_data_to_np(path: Path):
    data = torch.load(path)
    data_no_na = data[~torch.isnan(data[:, 0])]
    data_np = data_no_na.detach().numpy()
    return data_np


def split_data_reproduce(
    X, y: np.array, train_size=0.7, test_size=0.5, random_state: int = 42
):

    X_train, X_, y_train, y_ = train_test_split(
        X, y, train_size=train_size, random_state=random_state
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_, y_, test_size=test_size, random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


# convert x to tensor
def evaluate(y, y_pred):
    evals = {
        "f1_score": f1_score(y, y_pred),
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred),
        "recall": recall_score(y, y_pred),
        "roc_auc": roc_auc_score(y, y_pred),
        "confusion_matrix": confusion_matrix(y, y_pred),
    }
    return evals


def dataloader_to_numpy(path: Path, data_loader=AATrialDataModule):
    data = data_loader(path)
    data.prepare_data()
    X = data.data[:][0].detach().numpy()
    y = data.data[:][1].detach().numpy()
    return X, y
