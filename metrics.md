# Model Metrics: Baseline vs Text-Enhanced

Both models trained on an identical 400/100 train/test split (`random_state=42`),
so the only variable that changes between Day 4 and Day 5 is the feature set.

## Day 4 — Baseline (structured CRM fields only, 20 features)

| Model | ROC-AUC | PR-AUC | Threshold (max F1) | F1 |
|---|---|---|---|---|
| Logistic Regression | 0.710 | 0.525 | 0.455 | 0.538 |
| Random Forest | 0.692 | 0.470 | 0.380 | 0.492 |

**Top features (Random Forest):** `last_meeting_outcome_Positive`, `last_meeting_outcome_Objection`,
`last_meeting_outcome_None-yet`, `source_Referral`, `annual_income_lakhs`

## Day 5 — Enhanced (structured fields + 4 LLM-extracted signals, 31 features)

| Model | ROC-AUC | PR-AUC | Threshold (max F1) | F1 |
|---|---|---|---|---|
| Logistic Regression | 0.767 | 0.603 | 0.579 | 0.636 |
| Random Forest | 0.775 | 0.599 | 0.561 | 0.564 |

**Top features (Random Forest):** `sentiment_Positive` (#1, 0.155), `last_meeting_outcome_Positive`,
`days_since_first_contact`, `source_Referral`, `sentiment_Neutral`

## Result

Adding the 4 LLM-extracted signals (sentiment, urgency, objection, product_interest) on top of the
same structured CRM fields improved PR-AUC from **0.525 → 0.603** (Logistic Regression) and
**0.470 → 0.599** (Random Forest) — roughly a 15% and 27% relative improvement respectively.

`sentiment_Positive` became the single most important feature in the Random Forest model,
ranking above every structured CRM field including income and investable surplus. This is
direct, quantitative evidence that unstructured RM notes contain real predictive signal that
structured CRM data alone misses — not just noise dressed up as extra columns.

Weaker LLM signals (`urgency`, `objection`) ranked low in feature importance, consistent with
their lower extraction accuracy against `hidden_note_tags.csv` (~47.8% for urgency vs ~73.2%
for sentiment). The model correctly learned to lean less on unreliable features rather than
being misled by them.

## Confusion Matrices

**Day 4 Logistic Regression** (threshold=0.455):

    [[62 15]
     [ 9 14]]

**Day 5 Logistic Regression** (threshold=0.579):

    [[70  7]
     [ 9 14]]

Same number of true positives caught (14), but false positives dropped from 15 to 7 —
the enhanced model is more precise at the same recall level.
