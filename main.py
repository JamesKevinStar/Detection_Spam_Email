import os
import mlflow
import kagglehub
import pandas as pd

from typing import List, Tuple
from mlflow import MlflowClient
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def download_dataset(kaggle_path: str) -> str:
    """
    Download a dataset from KaggleHub and return the path
    """
    path = kagglehub.dataset_download(kaggle_path)
    return path

def load_dataset(path: str) -> pd.DataFrame:
    '''
    Load the dataset into a dataframe
    '''
    csv_file = os.path.join(path, os.listdir(path)[0])
    df = pd.read_csv(csv_file)
    return df

def train_test(df: pd.DataFrame, 
               features: List[str], 
               label: str, 
               train_size: float, 
               random_state: int
               ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    '''
    Split a dataframe into test and train sets.
    '''
    X = df[features]
    Y = df[label]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = train_size, random_state = random_state)
    return X_train, X_test, Y_train, Y_test

if __name__ == "__main__":
    # Download data
    path = download_dataset("ssssws/spam-email-detection-dataset-clean-and-ml-ready")

    # Load data
    df = load_dataset(path)
    print(df.head(5))

    # Split train, test
    features = ["num_words", "num_links", "num_exclamation_marks", "sender_reputation_score",
                "num_recipients", "contains_money_terms", "contains_urgency_terms"]
    
    label = "label"

    X_train, X_test, Y_train, Y_test = train_test(df, features, label, train_size = 0.7, random_state = 290)

    # Define models

    # Polynomial Regression:
    params_pr = {
        "degree" : 2,
    }
    
    # Logistic Regression:
    params_lr = {
        "C" : 0.02,
        "solver" : "lbfgs"
    }

    # MLflow
    models = [
        (
            "Polynomial Regresion",
            Pipeline([
                ("poly_features", PolynomialFeatures(
                    degree = params_pr["degree"], 
                    include_bias = False
                )),
                ("lin_reg", LogisticRegression())
            ])
        ),
        (
            "Logistic Regresion",
            LogisticRegression(C = params_lr["C"], solver = params_lr["solver"])
        )
    ]

    best_model = None

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:8080") # Keeps the default value, allowing to change later if needed
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("Spam Detection")

    for model_name, model in models:
        with mlflow.start_run(run_name = model_name) as run:
            model.fit(X_train, Y_train)

            predictions = model.predict(X_test)
            report = classification_report(
                Y_test,
                predictions,
                output_dict = True,
            )

            score = report["weighted avg"]["f1-score"]

            if best_model is None or score > best_model["score"]:
                best_model = {
                    "name": model_name,
                    "run_id": run.info.run_id, 
                    "score": score
                }   

            mlflow.log_param("model_name", model_name)
            if model_name == "Polynomial Regresion":
                mlflow.log_param("degree", params_pr["degree"])
            elif model_name == "Logistic Regresion":
                mlflow.log_param("C", params_lr["C"])
                mlflow.log_param("solver", params_lr["solver"])

            mlflow.log_metric("accuracy", report["accuracy"])
            mlflow.log_metric("recall_class_0", report["0"]["recall"])
            mlflow.log_metric("recall_class_1", report["1"]["recall"])
            mlflow.log_metric("recall", report["weighted avg"]["recall"])
            mlflow.log_metric("f1_class_0", report["0"]["f1-score"])
            mlflow.log_metric("f1_class_1", report["1"]["f1-score"])
            mlflow.log_metric("f1", report["weighted avg"]["f1-score"])

            mlflow.sklearn.log_model(model, artifact_path = "model")

    if best_model is None:
        raise Exception("No model was trained. Please check the training process.")

    registered_model = mlflow.register_model(
        model_uri = "runs:/{}/model".format(best_model["run_id"]),
        name = "spam-email-classifier"
    )

    MlflowClient().set_registered_model_alias(
        name = "spam-email-classifier",
        alias = "champion",
        version = registered_model.version
    )

    print("Registered {} as champion\n(weighted F1: {})".format(
        best_model["name"], 
        round(best_model["score"], 3)
    ))