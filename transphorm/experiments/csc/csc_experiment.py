import comet_ml
import os
from pathlib import Path
import joblib
import structlog

import numpy as np
from dotenv import load_dotenv

from transphorm.model_components.model_modules import (
    CDLTrainer,
    CDLAnalyzer,
)
from transphorm.framework_helpers import setup_comet_experimet
from transphorm.preprocessors.loaders import AADataLoader

load_dotenv()


def define_search_space():

    sec = 100
    times = [5, 10, 20, 30, 45, 60]
    configs = {
        "algorithm": "bayes",
        "spec": {
            "maxCombo": 30,
            "objective": "minimize",
            "metric": "test_mse",
            "minSampleSize": 88,
            "retryLimit": 20,
            "retryAssignLimit": 0,
        },
        "parameters": {
            "n_atoms": [5, 10, 15, 20, 25],
            "n_times_atom": [sec * t for t in times],
            "reg": [0.01, 0.05, 0.1, 0.5, 1, 5, 10],
            "n_iter": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            "n_jobs": [-1],
            # solver_d_kwargs
            "maxiter": [500, 1000, 1500, 2000],
            "tol": [1e-3, 1e-4, 1e-5, 1e-6],
            "factr": [1e5, 1e6, 1e7, 1e8],
            "pgtol": [1e-3, 1e-4, 1e-5, 1e-6],
            "l1_ratio": [0.01, 0.05, 0.1, 0.5, 0.95, 0.99],
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


def build_model(exp):
    solver_d_kwargs = {
        "maxiter": exp.get_parameter("maxiter"),
        "tol": exp.get_parameter("tol"),
        "factr": exp.get_parameter("factr"),
        "pgtol": exp.get_parameter("pgtol"),
        "l1_ratio": exp.get_parameter("l1_ratio"),
    }
    params = {
        "n_atoms": exp.get_parameter("n_atoms"),
        "n_times_atom": exp.get_parameter("n_times_atom"),
        "reg": exp.get_parameter("reg"),
        "n_iter": exp.get_parameter("n_iter"),
        "n_jobs": exp.get_parameter("n_jobs"),
        "verbose": 4,
        "random_state": 42,
    }
    csc = CDLTrainer(**params, solver_d_kwargs=solver_d_kwargs)
    return csc


def compute_z_and_x_hat(model, data):
    z_hat = model.transform(data)
    x_hat = model.reconstruct(z_hat)
    return z_hat, x_hat


def run_optimizer(project_name, opt, loader, log, model_save_dir):
    exp_configs = experiment_configs(project_name)
    for exp in opt.get_experiments(**exp_configs):
        log.info(f"training {exp.name}")
        model = build_model(exp)
        model.fit_csc(loader.train)

        analyzer = CDLAnalyzer(model, loader)
        analyzer.compute_z_and_x_hat()
        analyzer.compute_mses()
        exp.log_metrics("test_mse", analyzer.test_mse)
        exp.log_metrics("train_mse", analyzer.train_mse)

        exp.log_curve(name="Objective Function", x=model.trainer.pobjective)
        exp.log_figure("Objective Function", analyzer.plot_pobjective())
        exp.log_figure("MSE Distribution", analyzer.plot_mse_distribution())
        exp.log_figure("MSE Boxplot", analyzer.mse_boxplot())
        exp.log_figure("MSE by Trial", analyzer.plot_mse_by_trial())
        exp.log_figure("Atoms", analyzer.plot_atoms())
        exp.log_figure(
            "Best and Worst Reconstructions",
            analyzer.plot_best_and_worst_reconstructions(),
        )


def main():
    load_dotenv()
    log = structlog.get_logger()
    PROJECT_NAME = "csc"
    FULL_RECORDING_PATH = Path(os.getenv("FULL_RECORDING_PATH"))
    MODEL_SAVE_DIR = Path("/projects/p31961/transphorm/models/csc")
    MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    COMET_API_KEY = os.getenv("COMET_API_KEY")
    log.info("loading data")
    loader = AADataLoader(
        FULL_RECORDING_PATH,
        butter_filter=True,
        weiner_filter=False,
        weiner_window_size=1000,
        smoothing=True,
        smoothing_window_size=250,
        down_sample=True,
        down_sample_factor=10,
    )
    loader.load_data()
    loader.prepare_data(shape_for_arhmm=False)
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
