import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import polars as pl
from pathlib import Path
import structlog
import torch
import ssm
from dotenv import load_dotenv
import comet_ml
from transphorm.framework_helpers import setup_comet_experimet
from typing import List, Optional
import torch
from torch import Tensor
from dynamax.utils.plotting import gradient_cmap
from transphorm.preprocessors.loaders import AADataLoader
from transphorm.analyzers import ARHMMAnalyzer
import polars as pl


# read data
def load_data(path: Path, loader: AADataLoader) -> (np.ndarray, np.ndarray):
    loader = loader(path)
    loader.load_data()
    loader.prepare_data()
    return loader.x, loader.labels


def define_search_space():
    configs = {
        "algorithm": "bayes",
        "spec": {
            "maxCombo": 30,
            "objective": "maximize",
            "metric": "lls_max",
            "minSampleSize": 88,
            "retryLimit": 20,
            "retryAssignLimit": 0,
        },
        "parameters": {
            "K": [3, 4, 5, 6, 7, 8, 9, 10],
            "D": [1],
            "M": [0, 1],
            "method": ["ar"],
            "transitions": [
                "standard",
                "constrained",
                "sticky",
                "recurrent",
                "recurrent_only",
                "nn_recurrent",
            ],
            "observations": ["gaussian", "poisson", "bernoulli", "softmax"],
            "num_iters": [10, 20, 30, 40],
        },
    }
    return configs


def experiment_configs(project_name):
    exp_configs = {
        "project_name": project_name,
        "auto_param_logging": True,
        "auto_metric_logging": True,
        "auto_histogram_weight_logging": True,
        "auto_histogram_gradient_logging": True,
        "auto_histogram_activation_logging": True,
        "display_summary_level": 0,
    }
    return exp_configs


def train_model(exp, x):
    model_params = {
        "K": exp.get_parameter("K"),
        "D": exp.get_parameter("D"),
        "M": exp.get_parameter("M"),
        "transitions": exp.get_parameter("transitions"),
        "observations": exp.get_parameter("observations"),
    }
    num_iters = exp.get_parameter("num_iters")
    model = ssm.HMM(**model_params)
    lls = model.fit(x, method="em", num_iters=num_iters)
    return model, lls


# analyze states
def run_optimizer(project_name, opt, x, labels, log, model_save_dir):
    exp_configs = experiment_configs(project_name)
    for exp in opt.get_experiments(**exp_configs):
        model, lls = train_model(exp, x)
        params = model.get_params()
        exp.log_parameters(params)
        exp.log_metric("lls_max", lls[-1])

        analyzer = ARHMMAnalyzer(model, lls, x, labels)
        analyzer.compute_metrics()
        exp.log_curve(lls)
        exp.log_figure("Mean State Durations", analyzer.plot_mean_state_duration())
        exp.log_figure("Example States", analyzer.plot_states())
        exp.log_table("mean_state_durations.csv", analyzer.agg_data)
        joblib.dump(model, model_save_dir / f"{exp.name}.joblib")


# log results


def main():
    load_dotenv()
    log = structlog.get_logger()
    FULL_RECORDING_PATH = Path(os.getenv("FULL_RECORDING_PATH"))
    MODEL_SAVE_DIR = Path("/projects/p31961/transphorm/models/arhmm")
    MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    COMET_API_KEY = os.getenv("COMET_API_KEY")
    log.info("loading data")
    x, labels = load_data(path=FULL_RECORDING_PATH, loader=AADataLoader)
    log.info("configuring optimizer")
    opt = comet_ml.Optimizer(config=define_search_space())
    run_optimizer(
        project_name=PROJECT_NAME,
        opt=opt,
        x=x,
        labels=labels,
        log=log,
        model_save_dir=MODEL_SAVE_DIR,
    )
    log.info("experiment complete")


if __name__ == "__main__":
    main()
