import os
import mlflow
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel

tracking_uri = os.getenv("MLFLOW_TRACKING_URI", 
                         "http://127.0.0.1:8080")

mlflow.set_tracking_uri(tracking_uri)
model = mlflow.sklearn.load_model("models:/spam-email-classifier@champion")

app = FastAPI()

class EmailFeatures(BaseModel):
    num_words: float
    num_links: float
    num_exclamation_marks: float
    sender_reputation_score: float
    num_recipients: float
    contains_money_terms: float
    contains_urgency_terms: float

@app.post("/predict")
def predict_email_spam(features: EmailFeatures):
    input_data = pd.DataFrame([features.model_dump()])
    prediction = model.predict(input_data)
    return {"prediction": int(prediction[0])}