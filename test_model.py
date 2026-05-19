import mlflow
import pandas as pd

from sklearn.metrics import accuracy_score
from main import download_dataset, load_dataset, train_test

def test_model_accuracy():
    '''
    Test that a model achieves at least 85% accuracy
    '''
    # Download data
    path = download_dataset("ssssws/spam-email-detection-dataset-clean-and-ml-ready")

    # Load data
    df = load_dataset(path)

    # Split train, test
    features = ["num_words", "num_links", "num_exclamation_marks", "sender_reputation_score",
                "num_recipients", "contains_money_terms", "contains_urgency_terms"]    
    label = "label"

    _, X_test, _, Y_test = train_test(df, features, label, train_size = 0.8, random_state = 783)

    mlflow.set_tracking_uri(uri = "http://127.0.0.1:8080/")

    # Obtain the model to test
    model_v = 1
    model_name = "Polynomial Regresion"
    model_uri = "models:/{}/{}".format(model_name, model_v)
    model = mlflow.sklearn.load_model(model_uri)

    Y_pred = model.predict(X_test)
    acc = accuracy_score(Y_test, Y_pred)

    assert acc >= 0.85, "Model accuracy too low: {:.2f}".format(acc)
