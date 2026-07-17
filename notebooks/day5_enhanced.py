import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    confusion_matrix,
    classification_report,
)

PLOTS_DIR = "notebooks/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

df = pd.read_csv("data/leads_v1_merged.csv")

# ---------------------------------------------------------------
# 1. Feature engineering: same structured fields as day4_baseline,
#    PLUS the 4 LLM-extracted signals (sentiment, urgency, objection,
#    product_interest) as new categorical features.
# ---------------------------------------------------------------
def count_products(val):
    if pd.isna(val):
        return 0
    return len([p.strip() for p in str(val).split(",") if p.strip()])

df["num_existing_products"] = df["existing_products"].apply(count_products)

numeric_features = [
    "age",
    "annual_income_lakhs",
    "investable_surplus_lakhs",
    "days_since_first_contact",
    "touchpoints_last_30d",
    "num_existing_products",
]

categorical_features = [
    "city_tier",
    "occupation",
    "source",
    "last_meeting_outcome",
    # new LLM-extracted signals:
    "sentiment",
    "urgency",
    "objection",
    "product_interest",
]

X = df[numeric_features + categorical_features].copy()
y = df["converted_within_30d"].copy()

# one-hot encode categoricals (same approach as day4_baseline)
X = pd.get_dummies(X, columns=categorical_features, drop_first=True)

print("Feature matrix shape:", X.shape)
print("Features used:", X.columns.tolist())

# ---------------------------------------------------------------
# 2. Train/test split, stratified because classes are imbalanced
#    (identical random_state=42 and split ratio to day4_baseline
#    so the comparison is apples-to-apples)
# ---------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
print("Train positive rate:", y_train.mean().round(3))
print("Test positive rate:", y_test.mean().round(3))

scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[numeric_features] = scaler.fit_transform(X_train[numeric_features])
X_test_scaled[numeric_features] = scaler.transform(X_test[numeric_features])

# ---------------------------------------------------------------
# 3. Train models
# ---------------------------------------------------------------
log_reg = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
log_reg.fit(X_train_scaled, y_train)

rf = RandomForestClassifier(
    n_estimators=300, max_depth=6, class_weight="balanced", random_state=42
)
rf.fit(X_train, y_train)

models = {
    "Logistic Regression": (log_reg, X_test_scaled),
    "Random Forest": (rf, X_test),
}

# ---------------------------------------------------------------
# 4. Evaluate: ROC-AUC, PR-AUC, confusion matrix, chosen threshold
# ---------------------------------------------------------------
results = {}

for name, (model, X_eval) in models.items():
    probs = model.predict_proba(X_eval)[:, 1]

    roc_auc = roc_auc_score(y_test, probs)
    pr_auc = average_precision_score(y_test, probs)

    precisions, recalls, thresholds = precision_recall_curve(y_test, probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
    best_idx = np.argmax(f1_scores[:-1])
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]

    preds_at_threshold = (probs >= best_threshold).astype(int)
    cm = confusion_matrix(y_test, preds_at_threshold)

    print(f"\n{'='*50}")
    print(f"{name}")
    print(f"{'='*50}")
    print(f"ROC-AUC: {roc_auc:.3f}")
    print(f"PR-AUC:  {pr_auc:.3f}  (day4 baseline was 0.525)")
    print(f"Chosen threshold (max F1): {best_threshold:.3f} (F1={best_f1:.3f})")
    print("Confusion matrix (rows=actual, cols=predicted):")
    print(cm)
    print(classification_report(y_test, preds_at_threshold, digits=3))

    results[name] = {
        "probs": probs,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "threshold": best_threshold,
        "cm": cm,
    }

# ---------------------------------------------------------------
# 5. Plots: ROC curves and PR curves for both models side by side
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res["probs"])
    axes[0].plot(fpr, tpr, label=f"{name} (AUC={res['roc_auc']:.3f})")

axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
axes[0].set_xlabel("False positive rate")
axes[0].set_ylabel("True positive rate")
axes[0].set_title("ROC curve (with LLM signals)")
axes[0].legend()

for name, res in results.items():
    precisions, recalls, _ = precision_recall_curve(y_test, res["probs"])
    axes[1].plot(recalls, precisions, label=f"{name} (AP={res['pr_auc']:.3f})")

axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].set_title("Precision-recall curve (with LLM signals)")
axes[1].legend()

plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/07_enhanced_roc_pr_curves.png", dpi=100)
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, (name, res) in zip(axes, results.items()):
    cm = res["cm"]
    ax.imshow(cm, cmap="Blues")
    ax.set_title(f"{name}\nthreshold={res['threshold']:.2f}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/08_enhanced_confusion_matrices.png", dpi=100)
plt.close()

print(f"\nPlots saved to {PLOTS_DIR}/")

# ---------------------------------------------------------------
# 6. Feature importance (random forest) -- check whether the new
#    LLM-derived signals actually rank as useful features or not
# ---------------------------------------------------------------
importances = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(
    ascending=False
)
print("\nRandom Forest feature importances (top 15):")
print(importances.head(15).round(3))
