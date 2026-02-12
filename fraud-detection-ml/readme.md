# Fraud Detection ML Project

## Project Overview
This project focuses on detecting fraudulent transactions using machine learning.  
The goal is to build a robust model capable of identifying suspicious activities in real-time financial transactions.

---

## Problem Statement
Fraudulent transactions cost businesses millions annually.  
This project aims to:
- Analyze transaction data
- Detect fraudulent patterns
- Build a machine learning model for prediction
- Deploy the model via API for real-time inference

---

## Dataset

The dataset used in this project is the "Credit Card Fraud Detection" dataset from Kaggle.

Download link:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Place the downloaded CSV file inside the `data/` folder before running the project.

- Contains credit card transactions (or banking transactions)
- Features include transaction amount, time, user details, etc.
- The dataset is **imbalanced**, with fraudulent transactions being rare
- Stored in the `data/` folder

---

## Folder Structure
fraud-detection-ml/
├── api/ # FastAPI endpoints for model prediction
│ └── app.py
├── data/ # Raw and processed datasets
├── notebooks/ # EDA, feature engineering, and experiments
│ └── eda.ipynb
├── src/ # Scripts for training, preprocessing, and prediction
│ ├── train.py
│ ├── preprocess.py
│ └── predict.py
├── requirements.txt # Project dependencies
├── README.md # Project documentation
└── .gitignore # Files/folders to ignore in Git

## How to Run the Project

### 1. Clone the repository
```bash
git clone https://github.com/srushtiselkar29-stack/fraud-detection-ml.git
cd fraud-detection-ml

