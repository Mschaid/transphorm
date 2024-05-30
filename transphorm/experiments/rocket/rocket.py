import comet_ml
from pathlib import Path
from sktime.classification.kernel_based import RocketClassifier
import logging
import structlog


from transphorm.framework_helpers.sk_helpers import *
from transphorm.framework_helpers.comet_helpers import *
from comet_ml.integration.sklearn import log_model
from sklearn.model_selection import train_test_split


def main():
    log = structlog.get_logger()
    PROJECT_NAME = "rocket_classifier_baseline"
    MAIN_PATH = Path("/Users/mds8301/Desktop/temp")
    data_path = MAIN_PATH / "dopamine_full_timeseries_array.pt"
    log.info(f"Loading data from {data_path}")
    experiment = setup_comet_experimet(
        project_name=PROJECT_NAME, workspace="transphorm"
    )

    data = load_py_data_to_np(data_path)
    log.info(f"Loaded data with shape {data.shape}")
    X = data[:, 1:]
    y = data[:, 0]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    rocket = RocketClassifier(num_kernels=1000, n_features_per_kernel=100)
    log.info(f"Fitting rocket classifier")
    rocket.fit(X_train, y_train)
    log.info(f"Rocket classifier fitted, predicting on test set")
    y_pred = rocket.predict(X_test)
    log.info("evaluating on test set")
    evals = evaluate(y_test, y_pred)
    params = rocket.get_params()
    log.info(f"params: {params}")
    score = rocket.score(X_test, y_test)
    log.info(f"score: {score}")
    experiment.log_metrics(evals)
    experiment.log_parameters(params)
    experiment.log_metric("score", score)
    log_model(rocket, experiment)
    log.info(f"logged model")


if __name__ == "__main__":
    main()
