from pathlib import Path
import pickle
import comet_ml
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from transphorm.framework_helpers.sk_helpers import load_py_data_to_np
from transphorm.model_components.data_objects import AATrialDataModule
from transphorm.framework_helpers import (
    dataloader_to_numpy,
    split_data_reproduce,
    log_evaluaton,
    setup_comet_experimet,
)
from sktime.classification.deep_learning.lstmfcn import LSTMFCNClassifier
import os
from comet_ml.integration.sklearn import log_model

import structlog
import joblib

def set_hypertune_configs():
    configs = {
        "algorithm": "bayes",
        "spec": {
            "maxCombo": 20,
            "objective": "maximize",
            "metric": "test_accuracy",
            "minSampleSize": 500,
            "retryLimit": 20,
            "retryAssignLimit": 0,
        },
        "parameters": {
            "epochs": [500, 1000, 1500, 2000],
            "dropout": [0.2, 0.4, 0.6, 0.8],
            "kernel_sizes": ["32, 15, 9", "16, 10, 6", "8, 5, 3"],
            "filter_sizes": ["128, 256, 128", "64, 128, 64"],
            "lstm_size": [2, 4, 6, 8, 10],
            "random_state": [42],
        },
    }
    return configs



def load_data(path: Path):
    X, y = dataloader_to_numpy(path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    return X_train, X_test, y_train, y_test


def train(exp, X_train, X_test, y_train, y_test):
    kernel_sizes = tuple(map(int, exp.get_parameter("kernel_sizes").split(",")))
    filter_sizes = tuple(map(int, exp.get_parameter("filter_sizes").split(",")))
    model_params = {
        "dropout": exp.get_parameter("dropout"),
        "kernel_sizes": kernel_sizes,
        "filter_sizes": filter_sizes,
        "lstm_size": exp.get_parameter("lstm_size"),
        "random_state": exp.get_parameter("random_state"),
    }
    model = LSTMFCNClassifier(**model_params)
    model.fit(X_train, y_train)

    # model_name = f"lstmfcn_{exp.get_key()}"
    # log_model(exp, model, model_name)
    return model



def run_optimizer(project_name, opt, X_train, X_test, y_train, y_test, log, model_save_dir):
    for exp in opt.get_experiments(project_name=project_name, auto_metric_logging=True):
        log.info(f"training {exp.name}")

        model = train(exp, X_train, X_test, y_train, y_test)
        params = model.get_params()
        exp.log_parameters(params)

        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        log_evaluaton(y= y_train, y_pred= train_pred, data_cat= 'train', exp= exp)
        log_evaluaton(y = y_test, y_pred = test_pred, data_cat = 'test', exp = exp)

        joblib.dump(model, model_save_dir/f"{exp.name}.joblib")
        exp.end()


def main():
    load_dotenv()
    log = structlog.get_logger()
    PROJECT_NAME = "lstmnfcn_bayes_tuning_5_day"
    MODEL_SAVE_DIR = Path("/projects/p31961/transphorm/models/aa_classifiers/sk_models")

    DATA_PATH = Path(os.getenv("5_DAY_DATA_PATH"))
    COMET_API_KEY = os.getenv("COMET_API_KEY")

    log.info("loading data")
    X_train, X_test, y_train, y_test = load_data(DATA_PATH)
    log.info('configuring optimizer')
    opt = comet_ml.Optimizer(config = set_hypertune_configs())



    run_optimizer(project_name=PROJECT_NAME,
                   opt=opt,
                   X_train=X_train,
                   X_test=X_test,
                   y_train=y_train, 
                   y_test=y_test, 
                   log=log, 
                   model_save_dir=MODEL_SAVE_DIR)
    log.info("experiment complete")

if __name__ == "__main__":
    main()
