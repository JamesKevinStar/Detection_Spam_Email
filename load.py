import mlflow
import pandas as pd

# Load models
mlflow.set_tracking_uri("http://127.0.0.1:8080/")

model_uri = "models:/spam-email-classifier@champion"

loaded_model = mlflow.sklearn.load_model(model_uri)

# Predict value
features = ["num_words", "num_links", "num_exclamation_marks", "sender_reputation_score",
            "num_recipients", "contains_money_terms", "contains_urgency_terms"]

sample = pd.DataFrame(
    [[19.0, 2.0, 0.0, 0.66, 23.0, 0.0, 0.0]],
    columns = features,
)

prediction = loaded_model.predict(sample)
print("Predicted value: {}".format(prediction)) # Value expected: 0