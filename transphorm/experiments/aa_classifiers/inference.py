from joblib import load
from transphorm.experiments.aa_classifiers.aa_lstmfcn_bayes_5_day import load_data
from dotenv import load_dotenv
import numpy as np
import logging
import structlog
from pathlib import Path
import os


def main():
    log = structlog.get_logger()
    load_dotenv()
    MODEL_PATH = Path(
        "/projects/p31961/transphorm/models/aa_classifiers/sk_models/appropriate_pizza_5708.joblib"
    )
    DATA_PATH = Path(os.getenv("DATA_PATH_5_DAY"))
    path_to_save = Path("/home/mds8301/data/gaby_data/over_day_5/eval_data")
    RECORDING_LENGTH = 4
    model = load(MODEL_PATH)

    X_train, X_test, y_train, y_test = load_data(
        DATA_PATH, recording_length=RECORDING_LENGTH
    )
    model.fit(X_train, y_train)

    log.info("running infernece")
    results = {
        "x_train": X_train,
        "x_test": X_test,
        "y_train": y_train,
        "y_train_pred": model.predict(X_train),
        "y_test": y_test,
        "y_test_pred": model.predict(X_test),
    }
    print(results["x_train"].shape)
    print(results["x_test"].shape)
    print(results["y_train"].shape)
    print(results["y_test"].shape)

    log.info("saving data")
    for k, v in results.items():
        np.save(path_to_save / f"{k}.npy", v)
    np.savez(path_to_save / "inference_results_4sec.npz", **results)

    model.save(
        path="/projects/p31961/transphorm/models/aa_classifiers/sk_models/appropriate_pizza_5708"
    )

    log.info("done")


if __name__ == "__main__":
    main()
