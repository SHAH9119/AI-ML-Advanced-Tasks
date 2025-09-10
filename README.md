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
