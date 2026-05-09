import mlflow

if __name__ == "__main__":
    # Load models
    mlflow.set_tracking_uri("http://127.0.0.1:8080/")

    model_v = 1
    model_name = "Polynomial Regresion"
    model_uri = "models:/{}/{}".format(model_name, model_v)

    loaded_model = mlflow.sklearn.load_model(model_uri)

    # Predict value
    Y = loaded_model.predict([[19.0, 2.0, 0.0, 0.66, 23.0, 0.0, 0.0]])
    print(Y) # Value expected: 0