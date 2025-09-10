import pandas as pd
from transformers import pipeline

# 1. Load dataset
df = pd.read_csv("tickets.csv")
print(df.head())

# 2. Define candidate labels (tags)
candidate_labels = [
    "billing issue",
    "technical support",
    "account access",
    "password reset",
    "feature request",
    "bug report",
    "general inquiry"
]

# 3. Zero-shot classification pipeline (using a strong model like bart-large-mnli or distilbart)
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# 4. Predict tags for each ticket
def predict_tags(text, labels, top_k=3):
    result = classifier(text, candidate_labels=labels, multi_label=False)
    scores = list(zip(result["labels"], result["scores"]))
    return sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]

# Example: test on first 5 tickets
for i, row in df.head(5).iterrows():
    text = row["text"]
    preds = predict_tags(text, candidate_labels, top_k=3)
    print(f"\nTicket: {text}")
    for tag, score in preds:
        print(f"  {tag}: {score:.3f}")

# 5. Apply to full dataset & store results
all_preds = []
for text in df["text"]:
    preds = predict_tags(text, candidate_labels, top_k=3)
    all_preds.append([tag for tag, score in preds])

df["predicted_tags"] = all_preds
df.to_csv("ticket_predictions.csv", index=False)
print("\nPredictions saved to ticket_predictions.csv")
