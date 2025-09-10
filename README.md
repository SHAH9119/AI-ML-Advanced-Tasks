# AI-ML Advanced Tasks

This repository contains two machine learning projects:

1. **Task 1 – AG News Topic Classifier (BERT)**  
   Fine-tunes BERT on the AG News dataset and provides a Streamlit web app for predictions.  

2. **Task 2 – Customer Churn Prediction Pipeline**  
   Builds a reusable ML pipeline with preprocessing, model training, hyperparameter tuning, and export.

---

## 🚀 Features
- **Task 1**:  
  - Fine-tunes **BERT (bert-base-uncased)** on AG News dataset  
  - Evaluates with accuracy, F1-score, confusion matrix  
  - Interactive Streamlit app for predictions  

- **Task 2**:  
  - Preprocessing with `StandardScaler` + `OneHotEncoder`  
  - End-to-end pipeline using `ColumnTransformer` and `Pipeline`  
  - Models: Logistic Regression, Random Forest  
  - Hyperparameter tuning with `GridSearchCV`  
  - Model export with `joblib`  

---

## 📊 Datasets
- **Task 1**: AG News dataset (auto-downloaded via `datasets` library).  
- **Task 2**: Telco Customer Churn dataset (CSV format).  

---

## 🛠️ How to Run

### 🔹 Task 1 – AG News (BERT)
1. Train the model:
   ```bash
   python TASK1main.py
