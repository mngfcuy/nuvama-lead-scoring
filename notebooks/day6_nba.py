import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

DATA_DIR = "data"


def count_products(val):
    if pd.isna(val):
        return 0
    return len([p.strip() for p in str(val).split(",") if p.strip()])


def assign_priority_tier(prob):
    if prob >= 0.6:
        return "High"
    elif prob >= 0.3:
        return "Medium"
    else:
        return "Low"


def recommend_action(row):
    """
    Rule-based Next-Best-Action. Combines model priority tier with the
    LLM-extracted signals (objection, sentiment, urgency) to produce a
    concrete, human-readable recommendation for the RM.
    """
    tier = row["priority_tier"]
    objection = row["objection"]
    sentiment = row["sentiment"]
    urgency = row["urgency"]

    # 1. A specific objection always gets addressed directly, regardless of tier --
    #    ignoring a known objection is worse than any other action.
    if objection == "Pricing":
        return "Address pricing concern (share cost breakdown / discount options)"
    if objection == "Trust":
        return "Send credibility material (track record, testimonials, compliance docs)"
    if objection == "Timing":
        return "Schedule follow-up for their stated timeline, don't push now"
    if objection == "Competitor":
        return "Send competitive comparison / differentiation pitch"

    # 2. No objection: escalate hot, urgent, positive leads to immediate action
    if tier == "High" and sentiment == "Positive" and urgency == "High":
        return "Call today - high intent, ready to convert"
    if tier == "High":
        return "Call this week - strong conversion likelihood"

    # 3. Medium tier: keep warm
    if tier == "Medium" and urgency == "High":
        return "Prioritize outreach this week despite medium score - urgency flagged"
    if tier == "Medium":
        return "Nurture - send relevant product info, follow up in 2 weeks"

    # 4. Low tier: deprioritize unless something flags it as worth a second look
    if tier == "Low" and sentiment == "Positive":
        return "Low score but positive sentiment - light-touch nurture, re-evaluate later"
    return "Deprioritize - low conversion likelihood, minimal recent engagement signal"


def main():
    df = pd.read_csv(
    os.path.join(DATA_DIR, "leads_v1_merged.csv"),
    keep_default_na=False,
    na_values=[""],
)
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
        "sentiment",
        "urgency",
        "objection",
        "product_interest",
    ]

    X = df[numeric_features + categorical_features].copy()
    y = df["converted_within_30d"].copy()
    X = pd.get_dummies(X, columns=categorical_features, drop_first=True)

    # Train on the FULL dataset here -- this is for scoring/deployment,
    # not evaluation (evaluation already happened in day5_enhanced.py).
    scaler = StandardScaler()
    X_scaled = X.copy()
    X_scaled[numeric_features] = scaler.fit_transform(X[numeric_features])

    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    model.fit(X_scaled, y)

    df["conversion_probability_model"] = model.predict_proba(X_scaled)[:, 1]
    df["priority_tier"] = df["conversion_probability_model"].apply(assign_priority_tier)
    df["recommended_action"] = df.apply(recommend_action, axis=1)

    out_cols = [
        "lead_id",
        "conversion_probability_model",
        "priority_tier",
        "sentiment",
        "urgency",
        "objection",
        "product_interest",
        "recommended_action",
    ]
    scored = df[out_cols].sort_values("conversion_probability_model", ascending=False)

    print("Priority tier distribution:")
    print(scored["priority_tier"].value_counts())
    print("\nRecommended action distribution:")
    print(scored["recommended_action"].value_counts())
    print("\nTop 10 highest-priority leads:")
    print(scored.head(10).to_string(index=False))

    out_path = os.path.join(DATA_DIR, "leads_scored_v1.csv")
    scored.to_csv(out_path, index=False)
    print(f"\nSaved scored + NBA-tagged leads to {out_path}")


if __name__ == "__main__":
    main()