print("Training Started")

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import os

def train_model():
  #Load Dataset
  df = pd.read_csv("data/creditcard.csv")

  df["Class"] = df["Class"].str.replace("'","").astype(int)
  
  X = df.drop("Class",axis=1)
  y = df["Class"]

  #Split
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=42, stratify=y )

  #Model
  model = XGBClassifier(
	n_estimators = 100,
	max_depth = 5,
	learning_rate = 0.1,
	random_state = 42,
	use_label_encoder = False,
	eval_metric = 'logloss'
	)

  model.fit(X_train, y_train)

  #Evaluation
  y_pred = model.predict(X_test)
  print(classification_report(y_test, y_pred))

  #create models folder if does not exist
  os.makedirs("models",exist_ok = True)

  #save model
  joblib.dump(model, "models/xgboost_fraud_model.pkl")
  print("Model Saved successfully")
 
if __name__ == "__main__":
   train_model()
