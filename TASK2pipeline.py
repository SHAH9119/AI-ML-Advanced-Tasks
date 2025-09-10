import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# 1. Load dataset
df = pd.read_csv("churn.csv")

print("Dataset shape:", df.shape)
print("Columns:", df.columns)

# Assume target column is "Churn" (Yes/No or 0/1)
target = "Churn"

# 2. Split features/labels
X = df.drop(columns=[target])
y = df[target]

# Convert Yes/No to 1/0 if necessary
if y.dtype == 'O':
    y = y.map({"Yes": 1, "No": 0})

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Identify column types
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

print("Numeric:", numeric_features)
print("Categorical:", categorical_features)

# 4. Preprocessing
numeric_transformer = Pipeline(steps=[
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# 5. Define models
log_reg = LogisticRegression(max_iter=1000)
rf = RandomForestClassifier(random_state=42)

# 6. Build pipeline
pipeline = Pipeline(steps=[("preprocessor", preprocessor),
                           ("classifier", log_reg)])  # default = Logistic Regression

# 7. Hyperparameter tuning
param_grid = [
    {
        "classifier": [log_reg],
        "classifier__C": [0.1, 1.0, 10],
        "classifier__solver": ["lbfgs", "liblinear"]
    },
    {
        "classifier": [rf],
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [None, 10, 20],
        "classifier__min_samples_split": [2, 5]
    }
]

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring="f1", n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)
print("Best CV score:", grid_search.best_score_)

# Save grid search report
with open("results/gridsearch_report.txt", "w") as f:
    f.write(str(grid_search.best_params_))
    f.write("\n")
    f.write(str(grid_search.best_score_))

# 8. Evaluate on test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print("Test Accuracy:", acc)
print("Test F1:", f1)

print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# 9. Export pipeline
joblib.dump(best_model, "models/churn_pipeline.pkl")
print("Pipeline saved to models/churn_pipeline.pkl")
