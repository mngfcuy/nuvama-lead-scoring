"""
generate_leads.py

Generates ~500 synthetic leads for the Nuvama RM prototype.

Run with a fixed seed -> always produces the identical dataset.
Usage: python generate_leads.py
Output: data/leads_v1.csv
"""

import os
import numpy as np
import pandas as pd

# Resolve paths relative to this script's location, not the current working
# directory -- so this works whether you run it from the repo root or from
# inside src/.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

# Fixed seed so the dataset is reproducible every time we run this.
SEED = 42
rng = np.random.default_rng(SEED)

N_LEADS = 500


def generate_ids(n):
    """Simple unique lead IDs: L00001, L00002, ..."""
    return [f"L{str(i).zfill(5)}" for i in range(1, n + 1)]


def generate_age(n):
    """
    Age 25-70, weighted so 35-55 is most common.
    We do this with a mixture: mostly draw from a normal centered at 45,
    clipped to the valid range, so the middle of the range is denser
    than the edges (matches the brief: "more leads in 35-55 range").
    """
    ages = rng.normal(loc=45, scale=10, size=n)
    ages = np.clip(ages, 25, 70)
    return ages.round().astype(int)


def generate_city_tier(n):
    """Categorical: Tier-1 most common, Tier-3 least."""
    return rng.choice(
        ["Tier-1", "Tier-2", "Tier-3"],
        size=n,
        p=[0.5, 0.35, 0.15],
    )


def generate_occupation(n):
    return rng.choice(
        ["Salaried", "Self-employed", "Business owner", "Retired", "Other"],
        size=n,
        p=[0.45, 0.20, 0.20, 0.10, 0.05],
    )


def generate_income(n):
    """
    Annual income in lakhs, roughly 10-500, NOT uniform.
    Real income distributions are right-skewed (many people earn modestly,
    a long tail earns a lot) -> use a lognormal distribution, then clip.
    """
    income = rng.lognormal(mean=3.6, sigma=0.7, size=n)
    income = np.clip(income, 10, 500)
    return income.round(1)


def generate_surplus(income):
    """
    Investable surplus should be CORRELATED with income, not independent.
    Why: someone earning more generally has more left over to invest, but
    it's not a fixed fraction -- people at the same income save differently
    (lifestyle, dependents, debt). So we take a fraction of income that
    itself varies per person (noise on the fraction), plus a bit of
    independent random variation.
    """
    n = len(income)
    savings_fraction = rng.beta(a=2, b=5, size=n) * 1.2
    surplus = income * savings_fraction
    surplus += rng.normal(0, 2, size=n)
    surplus = np.clip(surplus, 1, None)
    return surplus.round(1)


def generate_existing_products(n):
    """
    Subset of MF/PMS/AIF/Insurance/FD/Direct Equity/None.
    Most leads hold 0-2 products; 'None' should be common (they're leads,
    not clients yet, though some already invest elsewhere).
    """
    options = ["MF", "PMS", "AIF", "Insurance", "FD", "Direct Equity"]
    result = []
    for _ in range(n):
        num_products = rng.choice([0, 1, 2, 3], p=[0.35, 0.35, 0.20, 0.10])
        if num_products == 0:
            result.append("None")
        else:
            chosen = rng.choice(options, size=num_products, replace=False)
            result.append(", ".join(chosen))
    return result


def generate_source(n):
    return rng.choice(
        ["Referral", "Digital", "Cold-call", "Event", "Existing-client-family"],
        size=n,
        p=[0.25, 0.30, 0.20, 0.15, 0.10],
    )


def generate_days_since_first_contact(n):
    """0-180 days. Skew toward more recent leads (pipeline churns)."""
    days = rng.exponential(scale=45, size=n)
    days = np.clip(days, 0, 180)
    return days.round().astype(int)


def generate_touchpoints(n):
    """Calls+emails+meetings combined in last 30 days, 0-15."""
    touches = rng.poisson(lam=4, size=n)
    touches = np.clip(touches, 0, 15)
    return touches


def generate_last_meeting_outcome(n):
    return rng.choice(
        ["Positive", "Neutral", "Objection", "No-show", "None-yet"],
        size=n,
        p=[0.20, 0.30, 0.20, 0.10, 0.20],
    )


def main():
    ids = generate_ids(N_LEADS)
    age = generate_age(N_LEADS)
    city_tier = generate_city_tier(N_LEADS)
    occupation = generate_occupation(N_LEADS)
    income = generate_income(N_LEADS)
    surplus = generate_surplus(income)
    existing_products = generate_existing_products(N_LEADS)
    source = generate_source(N_LEADS)
    days_since_first_contact = generate_days_since_first_contact(N_LEADS)
    touchpoints_last_30d = generate_touchpoints(N_LEADS)
    last_meeting_outcome = generate_last_meeting_outcome(N_LEADS)

    df = pd.DataFrame({
        "lead_id": ids,
        "age": age,
        "city_tier": city_tier,
        "occupation": occupation,
        "annual_income_lakhs": income,
        "investable_surplus_lakhs": surplus,
        "existing_products": existing_products,
        "source": source,
        "days_since_first_contact": days_since_first_contact,
        "touchpoints_last_30d": touchpoints_last_30d,
        "last_meeting_outcome": last_meeting_outcome,
    })

    print(df.head(10))
    print("\nShape:", df.shape)
    print("\nIncome describe:\n", df["annual_income_lakhs"].describe())
    print("\nSurplus describe:\n", df["investable_surplus_lakhs"].describe())

    out_path = os.path.join(DATA_DIR, "leads_v1_structured.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
