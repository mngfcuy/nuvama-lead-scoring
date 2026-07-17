"""
generate_label.py

Generates the converted_within_30d label from a hand-designed "hidden ground
truth" formula, per docs_design_spec.md.

IMPORTANT (per assignment rules, section 4.2):
Once this file is finalized and the dataset is generated, treat this formula
as a black box during modeling. Do not re-open this file while building
train.py / doing EDA-driven feature engineering -- that would be a form of
leakage (you'd be reverse-engineering your own answer key). Only come back
to this file during failure analysis, to explain *why* a prediction was wrong.

Run standalone for testing: python generate_label.py
"""

import os
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

SEED = 44  # independent seed again
rng = np.random.default_rng(SEED)


SOURCE_WARMTH = {
    "Existing-client-family": 1.8,
    "Referral": 1.5,
    "Event": 0.5,
    "Digital": 0.0,
    "Cold-call": -1.0,
}

OUTCOME_SCORE = {
    "Positive": 1.6,
    "Neutral": 0.3,
    "No-show": -0.6,
    "Objection": -1.0,
    "None-yet": 0.0,
}

SENTIMENT_SCORE = {
    "Positive": 1.2,
    "Neutral": 0.0,
    "Negative": -1.1,
}

W_SOURCE = 1.0
W_SURPLUS = 1.3
W_OUTCOME = 1.2
W_TOUCH = 0.7
W_SENTIMENT = 1.0
W_RECENCY = -0.6
W_PRODUCT = 0.5
NOISE_STD = 1.4


def normalize_surplus(surplus):
    return np.clip(surplus / 60.0, 0, 1)


def touchpoint_effect(touchpoints, last_meeting_outcome):
    base = np.clip(touchpoints / 10.0, 0, 1.5)
    if last_meeting_outcome == "Objection":
        return base * 0.4
    return base


def recency_penalty(days_since_first_contact):
    return np.clip(days_since_first_contact / 180.0, 0, 1)


def product_crosssell(existing_products):
    if existing_products == "None":
        return 0.0
    held = [p.strip() for p in existing_products.split(",")]
    if "MF" in held or "Insurance" in held:
        return 1.0
    return 0.4


def compute_latent(row):
    latent = (
        W_SOURCE * SOURCE_WARMTH[row["source"]]
        + W_SURPLUS * normalize_surplus(row["investable_surplus_lakhs"])
        + W_OUTCOME * OUTCOME_SCORE[row["last_meeting_outcome"]]
        + W_TOUCH * touchpoint_effect(row["touchpoints_last_30d"], row["last_meeting_outcome"])
        + W_SENTIMENT * SENTIMENT_SCORE[row["note_sentiment"]]
        + W_RECENCY * recency_penalty(row["days_since_first_contact"])
        + W_PRODUCT * product_crosssell(row["existing_products"])
        + rng.normal(0, NOISE_STD)
    )
    return latent


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def generate_labels(df, midpoint_shift=None, calibrate=True):
    latents = df.apply(compute_latent, axis=1).values

    if calibrate:
        best_shift, best_diff = None, None
        for shift in np.arange(-3, 5, 0.1):
            probs = sigmoid(latents - shift)
            positive_rate = probs.mean()
            diff = abs(positive_rate - 0.20)
            if best_diff is None or diff < best_diff:
                best_diff = diff
                best_shift = shift
        midpoint_shift = best_shift
        print(f"Calibrated midpoint_shift = {midpoint_shift:.2f} "
              f"(expected positive rate ~{sigmoid(latents - midpoint_shift).mean():.3f})")

    probabilities = sigmoid(latents - midpoint_shift)
    converted = (rng.random(len(df)) < probabilities).astype(int)
    return converted, probabilities, midpoint_shift


if __name__ == "__main__":
    structured_path = os.path.join(DATA_DIR, "leads_v1_structured.csv")
    tags_path = os.path.join(DATA_DIR, "hidden_note_tags.csv")

    df = pd.read_csv(structured_path, keep_default_na=False, na_values=[""])
    tags = pd.read_csv(tags_path)
    df = df.merge(tags[["lead_id", "note_sentiment"]], on="lead_id")

    converted, probabilities, shift = generate_labels(df)
    df["converted_within_30d"] = converted

    print("\nActual positive rate:", df["converted_within_30d"].mean().round(3))
    print("\nConversion rate by last_meeting_outcome:")
    print(df.groupby("last_meeting_outcome")["converted_within_30d"].mean().round(3))
    print("\nConversion rate by source:")
    print(df.groupby("source")["converted_within_30d"].mean().round(3))
    print("\nConversion rate by note_sentiment:")
    print(df.groupby("note_sentiment")["converted_within_30d"].mean().round(3))
    df["conversion_probability"] = probabilities
    labels_out = df[["lead_id", "conversion_probability", "converted_within_30d"]]
    labels_out.to_csv(os.path.join(DATA_DIR, "labels_v1.csv"), index=False)
    print(f"\nSaved {len(labels_out)} labels to data/labels_v1.csv")
