import os
import mlflow
import kagglehub
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def download_dataset(kaggle_path: str):
    """
    Download a dataset from KaggleHub and return the path
    """
    path = kagglehub.dataset_download(kaggle_path)
    return path

def load_dataset(path: str):
    '''
    Load the dataset into a dataframe
    '''
    csv_file = os.path.join(path, os.listdir(path)[0])
    df = pd.read_csv(csv_file)
    return df

if __name__ == "__main__":
    # Download data
    path = download_dataset("ssssws/spam-email-detection-dataset-clean-and-ml-ready")

    # Load data
    df = load_dataset(path)
    print(df.head(5))

    # Split train, test
    features = ["num_words", "num_links", "num_exclamation_marks", "sender_reputation_score",
                "num_recipients", "contains_money_terms", "contains_urgency_terms"]
    
    X = df[features]
    Y = df["label"]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = 0.7, random_state = 290)

    # Define models

    # Polynomial Regression:
    pr = PolynomialFeatures(degree = 2)

    params_pr = {
        "degree" : 2,
    }

    pr = Pipeline([
        ("poly_features", PolynomialFeatures(degree = params_pr["degree"], include_bias = False)),
        ("lin_reg", LogisticRegression())
    ])
    pr.fit(X_train, Y_train)

    predict_pr = pr.predict(X_test)
    report_pr = classification_report(Y_test, predict_pr, output_dict = True)

    # Logistic Regression:
    params_lr = {
        "C" : 0.02,
        "solver" : "lbfgs"
    }

    lr = LogisticRegression(**params_lr)
    lr.fit(X_train, Y_train)

    predict_lr = lr.predict(X_test)
    report_lr = classification_report(Y_test, predict_lr, output_dict = True)

    # MLflow
    models = [
        (
            "Polynomial Regresion",
            Pipeline([
                ("poly_features", PolynomialFeatures(degree = params_pr["degree"], include_bias = False)),
                ("lin_reg", LogisticRegression())
            ]),
            (X_train, Y_train),
            (X_test, Y_test)
        ),
        (
            "Logistic Regresion",
            LogisticRegression(C = 0.02, solver = "lbfgs"),
            (X_train, Y_train),
            (X_test, Y_test)
        )
    ]

    reports = [report_pr, report_lr]

    mlflow.set_tracking_uri(uri = "http://127.0.0.1:8080/")
    mlflow.set_experiment("Spam Detection")

    for i, items in enumerate(models):
        model_name = items[0]
        model = items[1]
        report = reports[i]

        with mlflow.start_run(run_name = model_name):
            model.fit(X_train, Y_train) # Load trained models
            mlflow.log_param("model_name", model_name)
            mlflow.log_metric("accuracy", report["accuracy"])
            mlflow.log_metric("recall_class_0", report["0"]["recall"])
            mlflow.log_metric("recall_class_1", report["1"]["recall"])
            mlflow.log_metric("recall", report["weighted avg"]["recall"])
            mlflow.log_metric("f1_class_0", report["0"]["f1-score"])
            mlflow.log_metric("f1_class_1", report["1"]["f1-score"])
            mlflow.log_metric("f1", report["weighted avg"]["f1-score"])
            mlflow.sklearn.log_model(model, model_name)