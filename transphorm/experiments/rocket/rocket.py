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
    # MAIN_PATH = Path("/Users/mds8301/Desktop/temp")
    MAIN_PATH = Path(
        "/home/mds8301/Gaby_raw_data/processed_full_recording_unlabled_data"
    )  # quest
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
        X, y, test_size=0.3, random_state=42
    )

    rocket = RocketClassifier(
        num_kernels=30000,
        n_features_per_kernel=10,
        rocket_transform="minirocket",
        n_jobs=-1,
        random_state=42,
    )

    log.info(f"Fitting rocket classifier")
    rocket.fit(X_train, y_train)
    log.info(f"Rocket classifier fitted, predicting on test set")
    log.info("evaluating on test set")
    evals = evaluate(
        y_train=y_train,
        y_train_pred=rocket.predict(X_train),
        y_test=y_test,
        y_test_pred=rocket.predict(X_test),
    )
    params = rocket.get_params()
    experiment.log_metrics(evals)
    experiment.log_parameters(params)
    log_model(model=rocket, experiment=experiment)
    log.info("Experiment logged to Comet")


if __name__ == "__main__":
    main()
