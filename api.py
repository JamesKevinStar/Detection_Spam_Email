import mlflow
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel

mlflow.set_tracking_uri("http://127.0.0.1:8080")
model = mlflow.sklearn.load_model("models:/spam-email-classifier@champion")