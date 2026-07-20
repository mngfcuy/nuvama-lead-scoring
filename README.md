# RM Buddy — Lead Prioritization & Next-Best-Action Prototype

## The problem

Relationship Managers (RMs) at Nuvama manage hundreds of leads at once and have
to decide, every morning, which ones to call first. Today that's mostly gut
feel — inconsistent across RMs and not backed by data.

This prototype takes a list of leads (structured CRM fields + free-text call
notes) and produces, for each one: a conversion probability, a set of
structured signals extracted from the RM's notes (sentiment, urgency,
objection, product interest), and a recommended Next-Best-Action — all
explainable enough that an RM who's never used a model can trust it.

Everything here is trained and evaluated on **synthetic data only** —
no real client data was used, by design.

## How to run it (5 commands)

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python3 src/generate_leads.py        # generates data/leads_v1.csv
python3 src/src/extract_signals.py   # LLM signal extraction -> data/extracted_signals_v1.csv
python3 notebooks/day6_nba.py        # trains on full data, scores + ranks -> data/leads_scored_v1.csv
```

Then, to view the dashboard:
```bash
python3 -m uvicorn app:app --reload
```
and open `http://localhost:8000`.

## Folder structure
## Results summary

Adding 4 LLM-extracted signals (sentiment, urgency, objection, product interest)
on top of structured CRM fields alone improved PR-AUC from **0.525 → 0.603**
(Logistic Regression). `sentiment_Positive` became the single most important
feature in the Random Forest model — ahead of income and portfolio size —
which is direct evidence that RM notes carry real predictive signal that
structured CRM data misses on its own.

Full metrics: see [`metrics.md`](metrics.md).
Full failure analysis (including a systematic blind spot on leads with no
decisive first meeting yet): see [`failure_analysis.md`](failure_analysis.md).

## Known limitations

- `urgency` and `objection` extraction accuracy (~48% and lower) is weaker
  than `sentiment` (~73%) against the hidden ground-truth tags — the model
  correctly learns to lean on them less, but this means some real signal
  in those two fields is likely being underused.
- The model has a systematic blind spot on leads where
  `last_meeting_outcome = None-yet` — see failure_analysis.md for detail.
- Synthetic data, however realistic, cannot capture everything a real
  sales relationship involves — some genuine unpredictability is expected
  and shows up in the failure analysis.
