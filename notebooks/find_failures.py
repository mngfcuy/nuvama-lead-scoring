import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("data/leads_v1_merged.csv")
notes_df = pd.read_csv("data/rm_notes_v1.csv")
df = df.merge(notes_df[["lead_id", "rm_notes"]], on="lead_id", how="left")

def count_products(val):
    if pd.isna(val):
        return 0
    return len([p.strip() for p in str(val).split(",") if p.strip()])

df["num_existing_products"] = df["existing_products"].apply(count_products)

numeric_features = [
    "age", "annual_income_lakhs", "investable_surplus_lakhs",
    "days_since_first_contact", "touchpoints_last_30d", "num_existing_products",
]
categorical_features = [
    "city_tier", "occupation", "source", "last_meeting_outcome",
    "sentiment", "urgency", "objection", "product_interest",
]

X = df[numeric_features + categorical_features].copy()
y = df["converted_within_30d"].copy()
X = pd.get_dummies(X, columns=categorical_features, drop_first=True)

# SAME split as day5_enhanced.py - identical random_state=42 guarantees
# identical train/test membership, so indices line up with df
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
probs = model.predict_proba(X_test)[:, 1]

results = pd.DataFrame({
    "lead_id": df.loc[X_test.index, "lead_id"].values,
    "true_label": y_test.values,
    "predicted_prob": probs,
    "sentiment": df.loc[X_test.index, "sentiment"].values,
    "urgency": df.loc[X_test.index, "urgency"].values,
    "objection": df.loc[X_test.index, "objection"].values,
    "last_meeting_outcome": df.loc[X_test.index, "last_meeting_outcome"].values,
    "touchpoints_last_30d": df.loc[X_test.index, "touchpoints_last_30d"].values,
    "rm_notes": df.loc[X_test.index, "rm_notes"].values,
})

# biggest misses: predicted high but did NOT convert (false positives)
false_positives = results[results["true_label"] == 0].sort_values("predicted_prob", ascending=False).head(10)
# biggest misses: predicted low but DID convert (false negatives)
false_negatives = results[results["true_label"] == 1].sort_values("predicted_prob", ascending=True).head(10)

print("=" * 60)
print("TOP FALSE POSITIVES (predicted high, actually did NOT convert)")
print("=" * 60)
print(false_positives.to_string(index=False))

print("\n" + "=" * 60)
print("TOP FALSE NEGATIVES (predicted low, actually DID convert)")
print("=" * 60)
print(false_negatives.to_string(index=False))

false_positives.to_csv("notebooks/false_positives.csv", index=False)
false_negatives.to_csv("notebooks/false_negatives.csv", index=False)
print("\nSaved to notebooks/false_positives.csv and notebooks/false_negatives.csv")
