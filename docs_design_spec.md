# Design Spec: Synthetic Label Generation

This document is the "hidden ground truth" design. Once `generate_leads.py` is
written and run, we treat the label as a black box for modeling purposes —
this file is our own record of what we baked in, for later explanation
(failure analysis, Day 10 demo).

## Strong drivers (real causal effect on conversion)

| Feature | Effect | Reasoning |
|---|---|---|
| `source` | Referral / Existing-client-family = warm, Cold-call = cold | Warm intros carry pre-existing trust |
| `investable_surplus_lakhs` | Positive, roughly linear after normalization | More money to deploy = more likely to act, but not the only factor |
| `last_meeting_outcome` | Positive >> Neutral > No-show ≈ Objection(negative) | Single strongest behavioral signal |
| `touchpoints_last_30d` | Positive, but interacts with outcome | High touchpoints + Objection ≠ high touchpoints + Positive |
| notes sentiment/urgency (hidden tag, not raw text) | Positive sentiment/high urgency = higher | This is what the LLM extractor is supposed to recover later |
| `days_since_first_contact` | Mild negative (staleness) | Leads sitting too long without movement cool off |
| `existing_products` | Mild positive if holding MF/Insurance (cross-sell warmth), neutral if None | Existing product holders are pre-qualified investors |

## Deliberate noise (included in data, ~no causal effect on label)

- `age`
- `city_tier`
- `occupation`
- `annual_income_lakhs` (redundant with surplus, mild/no independent effect)

Including noise features on purpose is a defensible design choice — real
datasets always have irrelevant columns, and a model correctly learning to
ignore them is itself a valid finding for the failure analysis / EDA writeup.

## One deliberate interaction

`touchpoints_last_30d` × `last_meeting_outcome`: touchpoints only help if the
last outcome isn't "Objection". This is the only interaction term for v1 —
keeping the rest additive keeps the formula explainable.

## Formula shape

```
latent = (
      w_source   * source_warmth[source]
    + w_surplus  * normalized(investable_surplus_lakhs)
    + w_outcome  * outcome_score[last_meeting_outcome]
    + w_touch    * touchpoint_effect(touchpoints_last_30d, last_meeting_outcome)
    + w_notes    * notes_sentiment_score        # hidden tag from note generation
    + w_recency  * recency_penalty(days_since_first_contact)
    + w_product  * product_crosssell[existing_products]
    + noise      # small random term, models irreducible uncertainty
)

probability = sigmoid(latent - midpoint_shift)
converted_within_30d = bernoulli_sample(probability)
```

`midpoint_shift` is tuned last, after weights are picked, purely to land the
overall positive rate in the 15-25% range.

## Weights (v1 — will tune after first generation run)

To be filled in once we write the actual generation code and check the
resulting positive rate.
