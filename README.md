# AI-ML Advanced Tasks

This repository contains multiple machine learning projects covering text classification, ML pipelines, and zero-shot learning.

---

## 📌 Tasks

1. **Task 1 – AG News Topic Classifier (BERT)**  
   Fine-tunes BERT on the AG News dataset and provides a Streamlit web app for predictions.  

2. **Task 2 – Customer Churn Prediction Pipeline**  
   Builds a reusable ML pipeline with preprocessing, model training, hyperparameter tuning, and export.  

3. **Task 5 – Automatic Ticket Tagging (Zero-Shot Classification)**  
   Uses Hugging Face's zero-shot classification to automatically assign support tags (e.g., *billing issue*, *technical support*) to customer support tickets without explicit training.  

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

- **Task 5**:  
  - Uses `facebook/bart-large-mnli` for zero-shot text classification  
  - Predicts top-k tags for each support ticket  
  - Exports results to `ticket_predictions.csv`  

---

## 📊 Datasets
- **Task 1**: AG News dataset (auto-downloaded via `datasets` library).  
- **Task 2**: Telco Customer Churn dataset (CSV format).  
- **Task 5**: `tickets.csv` containing customer support tickets with a `text` column.  

---

## 🛠️ How to Run

### 🔹 Task 1 – AG News (BERT)
1. Train the model:
   ```bash
   python TASK1main.py
