# Failure Analysis

Model: Logistic Regression (Day 5, enhanced feature set). Test set: 100 held-out leads
(random_state=42). Threshold: 0.579 (max F1).

## Pattern found: false negatives cluster on "None-yet" meeting outcomes

Across the 10 lowest-confidence false negatives, **every single one** has
`last_meeting_outcome = None-yet`. The model has learned that
`last_meeting_outcome_Positive` is a strong converting signal (it's the #2 most
important feature overall), but has no reliable signal for leads that haven't had
a decisive first meeting yet — even though a meaningful share of these leads still
convert. This is a genuine blind spot, not random noise.

## False Positives (predicted HIGH, did NOT convert)

| lead_id | predicted_prob | sentiment | last_meeting_outcome | Hypothesis |
|---|---|---|---|---|
| L00043 | 0.832 | Neutral | Neutral | Model over-weighted "asked to be contacted again" as a positive intent signal — it reads as engagement, but the note is actually non-committal ("think it over"). |
| L00134 | 0.774 | Positive | Positive | Strong positive sentiment + positive meeting outcome + SIP interest — genuinely looked like a good lead by every visible signal. Real-world outcome may hinge on something not captured in any feature (e.g. lost to a competitor, changed their mind). This is an honest model limitation, not a feature bug. |
| L00413 | 0.556 | Positive | Positive | High urgency + positive sentiment, but 0 touchpoints in the last 30 days — the model may be under-weighting recency/engagement here relative to sentiment. |
| L00458 | 0.522 | Neutral | Neutral | Has a "Timing" objection explicitly logged, but the model still predicted moderately high. Possible sign the `objection` feature isn't being weighted strongly enough — consistent with its lower extraction accuracy (~low agreement vs `hidden_note_tags.csv`). |
| L00111 | 0.449 | Neutral | Neutral | Note literally says "non-committal for now" — correctly borderline (near the 0.579 threshold), just tipped the wrong way. Reasonable near-miss, not a systemic error. |

## False Negatives (predicted LOW, DID convert)

| lead_id | predicted_prob | sentiment | last_meeting_outcome | Hypothesis |
|---|---|---|---|---|
| L00291 | 0.008 | Negative | None-yet | Negative sentiment + a logged Pricing objection — every visible signal points to "no." This lead converting suggests some conversions happen despite an initial rocky call, which no feature here captures (e.g. a later call not represented in this single note). |
| L00489 | 0.021 | Neutral | Objection | Explicit hesitation and a Timing objection, but still converted. Objection ≠ non-conversion; the model may be treating "objection present" as too strongly negative. |
| L00038 | 0.035 | Positive | None-yet | Positive sentiment and "ready to move forward soon" in the note, but the model predicted very low — likely dragged down hard by `last_meeting_outcome = None-yet`, overriding a fairly clear positive-language signal. |
| L00216 | 0.090 | Positive | None-yet | Positive sentiment, good engagement signals in the note, but no decisive meeting outcome yet. Same "None-yet" pattern. |
| L00080 | 0.281 | Neutral | Neutral | Generic, low-signal note ("no strong signal yet") that nonetheless converted. This is close to an unpredictable case — there may simply not be enough information in this note for any model to catch it. |

## What this tells us about the system

1. **`last_meeting_outcome = None-yet` is a real blind spot.** The model treats it almost
   like a negative signal by default, when in reality it should be treated closer to
   "unknown" — a meaningfully different thing. A fix worth trying: don't let the model rely as
   heavily on meeting outcome alone; give more weight to sentiment/urgency when meeting
   outcome is uninformative.
2. **The weaker LLM signals (`objection`, `urgency`) sometimes get overridden by stronger
   ones (`sentiment`, `last_meeting_outcome`) even when they contain the correct signal**
   (e.g. L00458, L00489). This lines up with their lower extraction accuracy against
   `hidden_note_tags.csv` — the model has learned, correctly, to trust them less, but
   that means it sometimes misses cases where they were actually right.
3. **Some failures look like genuine unpredictability**, not model error (L00134, L00291,
   L00080) — leads that looked good/bad by every available signal but converted/didn't
   convert anyway. In a real deployment, this is expected: the note + CRM fields captured
   here are not the entire picture of a real sales relationship.
