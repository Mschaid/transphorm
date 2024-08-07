from pathlib import Path
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


def build_model():
    lstmfcn = LSTMFCNClassifier(n_epochs=500)
    return lstmfcn


def main():
    MODEL_SAVE_DIR = Path()

    DATA_PATH = Path(os.getenv("TRIAL_DATA_PATH"))
    EXPERIMENT_NAME = "lstmfcn_aa_trial_v0"
    COMET_API_KEY = os.getenv("COMET_API_KEY")

    # set up comet experiment
    exp = comet_ml.Experiment(
        api_key=COMET_API_KEY,
        workspace="transphorm",
        project_name="aa-classifiers",
        experiment_key="aa_lstmfsn_v0",
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


# validate model

# save model` `


if __name__ == "__main__":

    main()
