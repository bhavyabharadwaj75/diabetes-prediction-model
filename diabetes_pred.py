
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc

df = pd.read_csv("diabetes.csv")
print(df.info())
print(df.describe()) 
# Missing Values Check
print("\nMissing Values:\n", df.isnull().sum())

plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()

sns.countplot(x="Outcome", data=df, palette="Set2")
plt.title("Class Distribution (0 = No Diabetes, 1 = Diabetes)")
plt.show()

X = df.drop(columns=["Outcome"]) 
y = df["Outcome"]  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
    "XGBoost": XGBClassifier(eval_metric="logloss", n_estimators=100, max_depth=5, random_state=42),  # FIXED
    "SVM": SVC(kernel="linear", probability=True, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=500, random_state=42)
}

param_grid = {"n_estimators": [50, 100, 150], "max_depth": [5, 10, 15]}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring="accuracy")  # CV = 3 for speed
grid_search.fit(X_train, y_train)
best_rf = grid_search.best_estimator_
models["Tuned Random Forest"] = best_rf

results = {}
roc_auc_scores = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"\n🔹 {name} Accuracy: {acc:.2f}")
    print(classification_report(y_test, y_pred))

    plt.figure(figsize=(5, 4))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    if y_pred_prob is not None:
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        roc_auc_scores[name] = roc_auc

        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, color="darkorange", label=f"AUC = {roc_auc:.2f}")
        plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
        plt.title(f"{name} - ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        plt.show()

plt.figure(figsize=(8, 5))
sns.barplot(x=list(results.keys()), y=list(results.values()), palette="viridis")
plt.title("Model Comparison (Accuracy)")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.show()

if roc_auc_scores:
    plt.figure(figsize=(8, 5))
    sns.barplot(x=list(roc_auc_scores.keys()), y=list(roc_auc_scores.values()), palette="coolwarm")
    plt.title("Model Comparison (ROC AUC Score)")
    plt.ylabel("AUC Score")
    plt.ylim(0, 1)
    plt.show()
