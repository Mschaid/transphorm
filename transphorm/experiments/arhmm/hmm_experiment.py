import os
from pathlib import Path


import comet_ml
import joblib
import ssm
import structlog

import numpy as np
from dotenv import load_dotenv


from transphorm.analyzers import ARHMMAnalyzer
from transphorm.framework_helpers import setup_comet_experimet
from transphorm.preprocessors.loaders import AADataLoader


# read data
def load_data(
    path: Path,
    loader: AADataLoader,
    down_sample: bool = True,
    down_sample_factor: int = 100,
    low_pass: bool = True,
) -> (np.ndarray, np.ndarray, np.ndarray):
    loader = loader(
        path,
        down_sample=down_sample,
        down_sample_factor=down_sample_factor,
        low_pass=low_pass,
    )
    loader.load_data()
    loader.prepare_data()

    return loader


def define_search_space():
    configs = {
        "algorithm": "bayes",
        "spec": {
            "maxCombo": 30,
            "objective": "maximize",
            "metric": "test_lls",
            "minSampleSize": 88,
            "retryLimit": 20,
            "retryAssignLimit": 0,
        },
        "parameters": {
            "K": [2, 3, 4, 5, 6],
            "D": [1],
            "M": [0, 1],
            "method": ["em"],
            "transitions": [
                "standard",
                "sticky",
            ],
            "observations": ["gaussian"],
            "num_iters": [5, 10, 20],
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
    lls = model.fit(x, method=exp.get_parameter("method"), num_iters=num_iters)
    return model, lls, model_params


# analyze states
def run_optimizer(project_name, opt, loader, log, model_save_dir):
    exp_configs = experiment_configs(project_name)
    for exp in opt.get_experiments(**exp_configs):
        log.info(f"training {exp.name}")
        model, lls, model_params = train_model(exp, loader.train)

        analyzer = ARHMMAnalyzer(model, lls, loader)
        analyzer.compute_metrics()
        exp.log_curve(name="Log Likehood", x=np.arange(len(lls)), y=lls)

        exp.log_parameters(model_params)
        exp.log_metric("train_lls", analyzer.training_metrics["train_lls"])
        exp.log_metric("test_lls", analyzer.training_metrics["test_lls"])
        log.info(f"training metrics {analyzer.training_metrics}")
        log.info(f"train_lls: {analyzer.training_metrics['train_lls']}")
        log.info(f"test_lls: {analyzer.training_metrics['test_lls']}")
        exp.log_figure("Log Likelihood", analyzer.plot_lls())
        exp.log_figure("Mean State Durations", analyzer.plot_mean_state_duration())
        exp.log_figure("Example States", analyzer.plot_states())
        exp.log_table("mean_state_durations.csv", analyzer.agg_data)
        joblib.dump(model, model_save_dir / f"{exp.name}.joblib")


def main():
    load_dotenv()
    log = structlog.get_logger()
    PROJECT_NAME = "hmm_longform_learning_partitioned"
    FULL_RECORDING_PATH = Path(os.getenv("FULL_RECORDING_PATH"))
    # FULL_RECORDING_PATH = Path(
    #     "/Users/mds8301/Desktop/temp/dopamine_full_timeseries_array.pt"
    # )
    MODEL_SAVE_DIR = Path("/projects/p31961/transphorm/models/arhmm")
    MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    COMET_API_KEY = os.getenv("COMET_API_KEY")
    log.info("loading data")
    loader = load_data(
        path=FULL_RECORDING_PATH, loader=AADataLoader, down_sample_factor=10
    )
    log.info("configuring optimizer")
    opt = comet_ml.Optimizer(config=define_search_space())
    run_optimizer(
        project_name=PROJECT_NAME,
        opt=opt,
        loader=loader,
        log=log,
        model_save_dir=MODEL_SAVE_DIR,
    )
    log.info("experiment complete")


if __name__ == "__main__":
    main()
