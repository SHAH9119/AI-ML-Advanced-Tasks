# AG News Topic Classifier (BERT)

This project trains a BERT model on the AG News dataset using Hugging Face's Transformers library,  
and provides a simple **Streamlit web app** for interactive predictions.

---

## 🚀 Features
- Fine-tunes **BERT (bert-base-uncased)** on the AG News dataset
- Evaluates performance using accuracy, F1-score, and confusion matrix
- Saves the trained model locally
- Streamlit app for entering custom news headlines and predicting their category

---

## 🛠️ How to Run

### 1. Train the Model
Run the training script to fine-tune BERT and save the model:

```bash

python main.py

# Task 2 – End-to-End ML Pipeline for Customer Churn Prediction

## 📌 Objective
Build a reusable and production-ready **machine learning pipeline** to predict customer churn using the **Telco Churn dataset**.  
This task demonstrates **data preprocessing, model training, hyperparameter tuning, and pipeline export** using **scikit-learn**.

---

## 📊 Dataset
- **Telco Customer Churn dataset** (CSV format).  
- Target column: `Churn` (Yes/No).  
- Contains both **numeric** (e.g., tenure, charges) and **categorical** features (e.g., gender, contract type).

---

## ⚙️ Approach
1. **Data Preprocessing**
   - Split into train/test (80/20).  
   - Numeric features → `StandardScaler`.  
   - Categorical features → `OneHotEncoder`.  

2. **Pipeline Construction**
   - Used `ColumnTransformer` + `Pipeline`.  
   - Models tested:
     - Logistic Regression
     - Random Forest  

3. **Hyperparameter Tuning**
   - Used `GridSearchCV` with 5-fold CV.  
   - Scoring metric: F1-score.  

4. **Evaluation**
   - Metrics: Accuracy, F1-score, Classification Report.  
   - Visualized Confusion Matrix.  

5. **Model Export**
   - Saved trained pipeline using `joblib` → `models/churn_pipeline.pkl`.  

---

## 🖥️ Project Structure
