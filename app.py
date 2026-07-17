import os
import pandas as pd
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from typing import Optional

DATA_PATH = "data/leads_scored_v1.csv"

app = FastAPI(title="RM Buddy API")


def load_leads():
    """Reads the scored leads CSV fresh on every call, so re-running
    day6_nba.py and refreshing the browser always shows the latest scores."""
    df = pd.read_csv(DATA_PATH, keep_default_na=False, na_values=[""])
    return df


@app.get("/api/summary")
def get_summary():
    df = load_leads()
    return {
        "total_leads": len(df),
        "tier_counts": df["priority_tier"].value_counts().to_dict(),
        "avg_conversion_probability": round(
            df["conversion_probability_model"].mean(), 3
        ),
    }


@app.get("/api/leads")
def get_leads(
    tier: Optional[str] = Query(None, description="Filter by priority tier: High/Medium/Low"),
    search: Optional[str] = Query(None, description="Filter by lead_id substring"),
):
    df = load_leads()

    if tier:
        df = df[df["priority_tier"].str.lower() == tier.lower()]
    if search:
        df = df[df["lead_id"].str.contains(search, case=False, na=False)]

    df = df.sort_values("conversion_probability_model", ascending=False)
    return df.to_dict(orient="records")


# serve the frontend
app.mount("/", StaticFiles(directory="static", html=True), name="static")