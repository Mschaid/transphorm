from pathlib import Path
import pickle
import comet_ml
from dotenv import load_dotenv
from transphorm.model_components.data_objects import AATrialDataModule
from transphorm.framework_helpers import (
    dataloader_to_numpy,
    split_data_reproduce,
    evaluate,
    setup_comet_experimet,
)
from sktime.classification.deep_learning.lstmfcn import LSTMFCNClassifier
import os
from sklearn.metrics import confusion_matrix

import joblib


def build_model():
    lstmfcn = LSTMFCNClassifier(n_epochs=2000, attention=True)
    return lstmfcn


def save_model(model_dir, experiment_name, model):
    joblib.dump(model, model_dir / f"{experiment_name}.pkl")


def main():
    MODEL_SAVE_DIR = Path(
        "/Users/mds8301/Development/transphorm/models/sk/aa_classifiers"
    )

    DATA_PATH = Path(os.getenv("TRIAL_DATA_PATH"))
    EXPERIMENT_NAME = "lstmfcn_aa_trial_v0"
    COMET_API_KEY = os.getenv("COMET_API_KEY")

    # set up comet experiment
    exp = comet_ml.Experiment(
        api_key=COMET_API_KEY,
        workspace="transphorm",
        project_name="aa-classifiers"
    )

    # load data from pytroch loaders
    X, y = dataloader_to_numpy(DATA_PATH)

    # spit data with reproduce
    X_train, X_val, X_test, y_train, y_val, y_test = split_data_reproduce(X, y)

    # define lstmfcn model
    model = build_model()

    # train model
    model.fit(X_train, y_train)

    # log training metrics
    y_train_pred = model.predict(X_train)
    with exp.train():
        metrics = evaluate(y_train, y_train_pred)
        exp.log_metrics(metrics)

    # log testing metrics,
    y_test_pred = model.predict(X_test)
    with exp.test():
        metrics = evaluate(y_test, y_test_pred)
        exp.log_metrics(metrics)

<<<<<<< HEAD
    # write confusion mat
    exp.log_confusion_matrix(y_test, y_test_pred)

    # save model`
    file_save_path = DATA_PATH /f"{EXPERIMENT_NAME}.pkl"
    with open(file_save_path, 'wb') as f:
        pickle.dump(model, f) 
=======
    exp.log_confusion_matrix(y_true=y_test, y_predicted=y_test_pred)
    save_model(MODEL_SAVE_DIR, EXPERIMENT_NAME, model)


# save model` `
>>>>>>> 54a7556b7590f8fc2a6063e9eff0a1e1ef6d7279

    exp.log_model(EXPERIMENT_NAME, file_save_path.as_posix)
    exp.end()

if __name__ == "__main__":
    main()
