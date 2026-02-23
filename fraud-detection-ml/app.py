from fastapi import FastAPI
from src.predict import predict
from pydantic import BaseModel
from typing import List 

app = FastAPI()

#Define input schema
class FraudInput(BaseModel):
  features: List[float]

@app.get("/")
def home():
  return {"message": "Fraud Detection API is running"}

@app.post("/predict")
def get_prediction(data : FraudInput):

  result = predict(data.features)
  return result       