from joblib import load
from transphorm.experiments.aa_classifiers.aa_lstmfcn_bayes_5_day import load_data
from dotenv import load_dotenv
import numpy as np
import logging
import structlog


def main():
    log = structlog.get_logger()
    load_dotenv()
    MODEL_PATH = Path(
        "/projects/p31961/transphorm/models/aa_classifiers/sk_models/accepted_plywood_8946.joblib"
    )
    DATA_PATH = Path(os.getenv("DATA_PATH_5_DAY"))
    path_to_save = Path("/home/mds8301/data/gaby_data/over_day_5/eval_data")
    model = load(MODEL_PATH)
    X_train, X_test, y_train, y_test = load_data(DATA_PATH)

    log.info("running infernece")
    results = {
        "x_train": X_train,
        "x_test": X_test,
        "y_train": y_train,
        "y_train_pred": model.predict(X_train),
        "y_test": y_test,
        "y_test_pred": model.predict(X_test),
    }

    log.info("saving data")

    np.savez(path_to_save, **results)
    log.info("done")
