import mlflow

if __name__ == "__main__":
    # Register models
    mlflow.set_tracking_uri("http://127.0.0.1:8080/")
    best_model_name = "Logistic Regresion"
    run_id = "ceed84ea0c6a40afb8be2b05e89137f3"
    model_uri = "runs:/{}/{}".format(run_id, best_model_name)

    mlflow.register_model(model_uri, best_model_name)