import comet_ml
import os
from pathlib import Path
import joblib
import structlog
import time
from matplotlib import pyplot as plt
import numpy as np
from dotenv import load_dotenv
import signal
from transphorm.model_components.model_modules import (
    CDLTrainer,
    CDLAnalyzer,
)
from transphorm.framework_helpers import setup_comet_experimet
from transphorm.preprocessors.loaders import AADataLoader

load_dotenv()


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException("Function call timed out")


def define_search_space(down_sample_factor):

    sample_rate = 1017
    sec = int(sample_rate / down_sample_factor)

    times = [15, 30, 45, 60, 90]
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
            "n_atoms": [3, 5],  # [5, 10, 15, 20, 25],
            "n_times_atom": [t * sec for t in times],
            "reg": [0.001, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5],
            "n_iter": [5, 6],  # [10, 20, 30, 40, 50],
            # solver_d_kwargs
            "maxiter": [100, 200],
            "tol": [1e-2],
            "factr": [1e7],
            "pgtol": [1e-5],
            "l1_ratio": [0.01, 0.05, 0.1],
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
        "n_jobs": 10,
        "verbose": 6,
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
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(60 * 5)
    for exp in opt.get_experiments(**exp_configs):
        start_time = time.time()
        log.info(f"Starting experiment: {exp.name}")

        log.info("Building model")
        model = build_model(exp)

        log.info("Fitting model")
        try:
            model.fit_csc(loader.train)

            log.info("Creating analyzer")
            analyzer = CDLAnalyzer(model, loader)

            log.info("Computing z and x hat")
            z_hat_train = model.transform(loader.train)
            x_hat_train = model.reconstruct(z_hat_train)
            analyzer.train_x_hat = x_hat_train
            z_hat_test = model.transform(loader.test)
            x_hat_test = model.reconstruct(z_hat_test)
            analyzer.test_x_hat = x_hat_test

            log.info("Computing MSEs")
            analyzer.compute_mses()
            log.info(f"Train MSE: {analyzer.train_mse}, Test MSE: {analyzer.test_mse}")

            exp.log_metric("test_mse", analyzer.test_mse)
            exp.log_metric("train_mse", analyzer.train_mse)
            exp.log_parameter("n_atoms", model.n_atoms)
            exp.log_parameter("n_times_atom", model.n_times_atom)
            exp.log_parameter("reg", model.reg)
            exp.log_parameter("n_iter", model.n_iter)
            exp.log_parameter("maxiter", model.solver_d_kwargs["maxiter"])
            exp.log_parameter("tol", model.solver_d_kwargs["tol"])
            exp.log_parameter("factr", model.solver_d_kwargs["factr"])
            exp.log_parameter("pgtol", model.solver_d_kwargs["pgtol"])
            exp.log_parameter("l1_ratio", model.solver_d_kwargs["l1_ratio"])

            # exp.log_curve(
            #     name="Objective Function",
            #     x=range(len(model.pobjective)),
            #     y=model.pobjective,
            # )
            # exp.log_figure("Objective Function", analyzer.plot_pobjective())
            # exp.log_figure("MSE Distribution", analyzer.plot_mse_distribution())
            # exp.log_figure("MSE Boxplot", analyzer.mse_boxplot())
            # exp.log_figure("MSE by Trial", analyzer.plot_mse_by_trial())
            # exp.log_figure("Atoms", analyzer.plot_atoms())
            # exp.log_figure(
            #     "Best and Worst Reconstructions",
            #     analyzer.plot_best_and_worst_reconstructions(),
            # )
            end_time = time.time()
            log.info(f"Experiment completed in {end_time - start_time:.2f} seconds")
            exp.end()
            plt.close("all")
        except TimeoutException as e:
            log.error(f"Error in experiment {exp.name}: {str(e)}")


def main():
    load_dotenv()
    log = structlog.get_logger()
    PROJECT_NAME = "csc_local"
    FULL_RECORDING_PATH = Path(os.getenv("FULL_RECORDING_PATH"))
    print(FULL_RECORDING_PATH)
    MODEL_SAVE_DIR = Path("/Users/mds8301/Desktop/csc")
    MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    COMET_API_KEY = os.getenv("COMET_API_KEY")
    DOWN_SAMPLE_FACTOR = 500
    log.info("loading data")
    loader = AADataLoader(
        FULL_RECORDING_PATH,
        butter_filter=True,
        weiner_filter=False,
        weiner_window_size=1000,
        smoothing=True,
        smoothing_window_size=250,
        down_sample=True,
        down_sample_factor=DOWN_SAMPLE_FACTOR,
    )
    loader.load_data()
    loader.prepare_data(shape_for_arhmm=False)
    log.info("configuring optimizer")

    search_space_config = define_search_space(DOWN_SAMPLE_FACTOR)
    opt = comet_ml.Optimizer(config=search_space_config)
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
