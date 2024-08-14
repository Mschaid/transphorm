import comet_ml
from pathlib import Path
from sktime.classification.kernel_based import RocketClassifier
import logging
import structlog


from transphorm.framework_helpers.sk_helpers import *
from transphorm.framework_helpers.comet_helpers import *
from comet_ml.integration.sklearn import log_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


def set_hypertune_configs():
    configs = {
        "algorithm": "bayes",
        "parameters": {
            "num_kernels": [1000, 10_000, 50_000, 100_000],
            "n_features_per_kernel": [10, 20, 50, 100],
            "rocket_transform": ["rocket", "minirocket"],
            "max_dilations_per_kernel": [8, 16, 24, 32, 64],
            "n_jobs": [-1],
            "random_state": [42],
        },
        "spec": {
            "metric": "f1_score_test",
            "objective": "maximize",
        },
    }
    return configs


def load_data(path, downsample_factor = 1):
    data = load_py_data_to_np(path)
    X = data[:, 1:]
    y = data[:, 0]
    if downsample_factor != 1:
        X = X[:, ::downsample_factor]
    return X, y


def train_rocket(exp, X_train, y_train, X_test, y_test):

    model_params = {
        "num_kernels": exp.get_parameter("num_kernels"),
        "n_features_per_kernel": exp.get_parameter("n_features_per_kernel"),
        "rocket_transform": exp.get_parameter("rocket_transform"),
        "max_dilations_per_kernel": exp.get_parameter("max_dilations_per_kernel"),
        "n_jobs": exp.get_parameter("n_jobs"),
        "random_state": exp.get_parameter("random_state"),
    }

    rocket = RocketClassifier(**model_params)
    model = rocket.fit(X_train, y_train)
    test_predict = model.predict(X_test)
    train_predict = model.predict(X_train)

    evals = evaluate(y_train, train_predict, y_test, test_predict)

    return model, evals


def run_optimizer(project_name, opt, X_train, X_test, y_train, y_test):
    for exp in opt.get_experiments(project_name=project_name, auto_metric_logging=True):

        model, evals = train_rocket(exp, X_train, y_train, X_test, y_test)

        f1_score_test = evals["f1_score_test"]

        params = model.get_params()
        exp.log_metrics(evals)
        exp.log_parameters(params)


def main():
    log = structlog.get_logger()
    PROJECT_NAME = "rocket_classifier_bayes_tuning"
   #  MAIN_PATH = Path("/Users/mds8301/Desktop/temp")
    MAIN_PATH = Path(
         "/home/mds8301/Gaby_raw_data/processed_full_recording_unlabled_data"
     )  # quest
    data_path = MAIN_PATH / "dopamine_full_timeseries_array.pt"
    DOWNSAMPLE_FACTOR = 100

    log.info(f"Loading data from {data_path}")

    """ load data"""

    X, y = load_data(data_path, downsample_factor= DOWNSAMPLE_FACTOR)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    """ set up experiments and tuning """
    log.info("running bayes tuning")
    opt = comet_ml.Optimizer(config=set_hypertune_configs())

    run_optimizer(
        project_name=PROJECT_NAME,
        opt=opt,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )

    # log_model(model=rocket, experiment=experiment)
    log.info("Tuning completed and logged to Comet")


if __name__ == "__main__":
    main()
