import os
import pandas as pd

DATA_DIR = "data"

leads = pd.read_csv(os.path.join(DATA_DIR, "leads_v1_structured.csv"))
notes = pd.read_csv(os.path.join(DATA_DIR, "hidden_note_tags.csv"))
labels = pd.read_csv(os.path.join(DATA_DIR, "labels_v1.csv"))

print("leads:", leads.shape)
print("notes:", notes.shape)
print("labels:", labels.shape)

# merge structured leads with notes on lead_id
merged = leads.merge(notes, on="lead_id", how="inner", validate="one_to_one")

# merge in labels (this brings in conversion_probability + converted_within_30d)
# labels already has note_sentiment logic baked into it via generate_label.py,
# but not the note_sentiment column itself, so no column clash here
merged = merged.merge(
    labels[["lead_id", "conversion_probability", "converted_within_30d"]],
    on="lead_id",
    how="inner",
    validate="one_to_one"
)

print("merged:", merged.shape)

if merged.shape[0] != leads.shape[0]:
    print(f"warning: row count changed from {leads.shape[0]} to {merged.shape[0]}, check for missing lead_ids")

out_path = os.path.join(DATA_DIR, "leads_v1.csv")
merged.to_csv(out_path, index=False)
print(f"saved {out_path}")