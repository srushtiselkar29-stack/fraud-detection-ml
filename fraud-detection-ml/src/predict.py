import joblib
import pandas as pd

#Load model
model = joblib.load("models/xgboost_fraud_model.pkl")

#Loading training columns
training_columns = pd.read_csv("data/creditcard.csv").drop("Class",axis=1).columns


def predict(features : list):

  #validate feature length
  if len(features) != len(training_columns):
    raise ValueError(f"Expected {len(training_columns)} features")

  # create dataframe with correct column names
  df = pd.DataFrame([features], columns = training_columns)   
  
  prediction = model.predict(df)[0]
  probability = model.predict_proba(df)[0][1]

  return {
    "Prediction" : int(prediction),
    "Fraud-probability" : float(probability)
    }      
  