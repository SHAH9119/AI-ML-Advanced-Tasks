import numpy as np
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoConfig,
                          AutoModelForSequenceClassification,
                          TrainingArguments, Trainer)
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load dataset
dataset = load_dataset("ag_news")
print(dataset)

# 2. Tokenizer + preprocess
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
label_names = dataset["train"].features["label"].names
num_labels = len(label_names)

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

tokenized = dataset.map(preprocess_function, batched=True)
tokenized = tokenized.rename_column("label", "labels")
tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

train_dataset = tokenized["train"]
eval_dataset = tokenized["test"]

# 3. Model config + model
id2label = {i: label_names[i] for i in range(num_labels)}
label2id = {label_names[i]: i for i in range(num_labels)}
config = AutoConfig.from_pretrained(model_name, num_labels=num_labels,
                                    id2label=id2label, label2id=label2id)
model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)

# 4. Training setup
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_steps=200,
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    return {"accuracy": acc, "f1": f1}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# 5. Train & evaluate
trainer.train()
metrics = trainer.evaluate()
print("Final Metrics:", metrics)

# 6. Extra evaluation
pred_output = trainer.predict(eval_dataset)
preds = np.argmax(pred_output.predictions, axis=-1)
labels = pred_output.label_ids

print(classification_report(labels, preds, target_names=label_names))

cm = confusion_matrix(labels, preds)
plt.figure(figsize=(7,6))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=label_names, yticklabels=label_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.savefig("results/confusion_matrix.png")
plt.show()

# 7. Save model + tokenizer
save_dir = "models/agnews-bert"
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
print(f"Model saved in {save_dir}")
