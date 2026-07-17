import os
import pandas as pd

DATA_DIR = "data"


def main():
    leads = pd.read_csv(os.path.join(DATA_DIR, "leads_v1_structured.csv"))
    signals = pd.read_csv(
        os.path.join(DATA_DIR, "extracted_signals_v1.csv"),
        keep_default_na=False,  # prevents pandas from reading the literal "None" as NaN
        na_values=[""],         # only truly empty cells count as missing
    )
    labels = pd.read_csv(os.path.join(DATA_DIR, "labels_v1.csv"))

    print(f"leads_v1_structured: {leads.shape}")
    print(f"extracted_signals_v1: {signals.shape}")
    print(f"labels_v1: {labels.shape}")

    # sanity check: confirm "None" survived correctly this time
    print("\nobjection value counts (should show 'None' as a real category, not NaN):")
    print(signals["objection"].value_counts(dropna=False))

    merged = leads.merge(signals, on="lead_id", how="left")
    merged = merged.merge(labels, on="lead_id", how="left")

    print(f"\nMerged shape: {merged.shape}")
    print(f"Rows with missing signals (no note extracted for that lead): {merged['sentiment'].isna().sum()}")

    # any lead with no note at all -> fall back to safe neutral defaults
    fill_values = {
        "sentiment": "Neutral",
        "urgency": "Medium",
        "objection": "None",
        "product_interest": "Unclear",
    }
    merged.fillna(fill_values, inplace=True)

    out_path = os.path.join(DATA_DIR, "leads_v1_merged.csv")
    merged.to_csv(out_path, index=False)
    print(f"\nSaved merged dataset to {out_path}")
    print(f"Final columns: {merged.columns.tolist()}")


if __name__ == "__main__":
    main()