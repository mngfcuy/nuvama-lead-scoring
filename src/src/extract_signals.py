import os
import sys
import json
import time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from anthropic import AnthropicBedrock

_debug_printed = False
DATA_DIR = "data"

# Bedrock model IDs are formatted differently from direct API model names.
# Confirm the exact string with whoever set up your AWS Bedrock access --
# it may need a region prefix like "us." depending on account configuration.
MODEL = "global.anthropic.claude-haiku-4-5-20251001-v1:0"
AWS_REGION = "ap-south-1" # change to whatever region your Bedrock access is granted in
MAX_WORKERS = 8  # concurrent API calls, keep modest to avoid rate limits

# Reads the Bedrock API key from the AWS_BEARER_TOKEN_BEDROCK environment variable
# automatically -- no need to hardcode it here. Set it in your shell first:
#   export AWS_BEARER_TOKEN_BEDROCK="your-key-here"
if not os.environ.get("AWS_BEARER_TOKEN_BEDROCK"):
    print(
        "ERROR: AWS_BEARER_TOKEN_BEDROCK is not set in this shell.\n"
        "Run this first, then re-run the script:\n"
        '  export AWS_BEARER_TOKEN_BEDROCK="your-bedrock-key-here"',
        file=sys.stderr,
    )
    sys.exit(1)

client = AnthropicBedrock(aws_region=AWS_REGION)

SYSTEM_PROMPT = """You extract structured signals from relationship manager (RM) meeting notes for a wealth management firm.

Given a note, output ONLY a JSON object, nothing else, no markdown fences, no preamble. The JSON must have exactly these 4 keys:

- "sentiment": one of "Positive", "Neutral", "Negative"
- "urgency": one of "High", "Medium", "Low"
- "objection": one of "Pricing", "Trust", "Timing", "Competitor", "None"
- "product_interest": one of "MF", "PMS", "AIF", "SIP", "Unclear"

Rules:
- "objection" should be "None" unless the note clearly expresses a specific concern matching one of the 4 categories.
- "product_interest" should be "Unclear" if no specific product is mentioned or implied.
- Base your answer only on what the note actually says, don't guess beyond the text."""

VALID_VALUES = {
    "sentiment": {"Positive", "Neutral", "Negative"},
    "urgency": {"High", "Medium", "Low"},
    "objection": {"Pricing", "Trust", "Timing", "Competitor", "None"},
    "product_interest": {"MF", "PMS", "AIF", "SIP", "Unclear"},
}

SAFE_DEFAULT = {
    "sentiment": "Neutral",
    "urgency": "Medium",
    "objection": "None",
    "product_interest": "Unclear",
}


def extract_one(lead_id, note_text):
    """Call the API for one note, parse and validate the JSON, fall back safely on any failure."""
    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=200,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": note_text}],
        )
        raw_text = response.content[0].text.strip()

        # defensive cleanup in case the model wraps the JSON in markdown fences anyway
        if raw_text.startswith("```"):
            raw_text = raw_text.strip("`")
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]
            raw_text = raw_text.strip()

        parsed = json.loads(raw_text)

        # validate every field is one of the allowed categories, replace with default if not
        result = {}
        for field, allowed in VALID_VALUES.items():
            value = parsed.get(field)
            result[field] = value if value in allowed else SAFE_DEFAULT[field]

        result["lead_id"] = lead_id
        result["extraction_status"] = "ok"
        return result

    except Exception as e:
        global _debug_printed
        if not _debug_printed:
            import traceback
            traceback.print_exc()
            _debug_printed = True
        # any failure (API error, bad JSON, timeout) falls back to safe defaults
        # rather than crashing the whole batch over one bad note
        result = dict(SAFE_DEFAULT)
        result["lead_id"] = lead_id
        result["extraction_status"] = f"failed: {type(e).__name__}"
        return result


def main():
    notes_df = pd.read_csv(os.path.join(DATA_DIR, "rm_notes_v1.csv"))
    print(f"Loaded {len(notes_df)} notes")

    results = []
    start = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(extract_one, row["lead_id"], row["rm_notes"]): row["lead_id"]
            for _, row in notes_df.iterrows()
        }
        completed = 0
        for future in as_completed(futures):
            results.append(future.result())
            completed += 1
            if completed % 50 == 0:
                print(f"  {completed}/{len(notes_df)} done...")

    elapsed = time.time() - start
    print(f"\nExtraction finished in {elapsed:.1f}s")

    extracted_df = pd.DataFrame(results)

    n_failed = (extracted_df["extraction_status"] != "ok").sum()
    print(f"Failed extractions (using safe defaults): {n_failed}/{len(extracted_df)}")
    if n_failed > 0:
        print(extracted_df[extracted_df["extraction_status"] != "ok"]["extraction_status"].value_counts())

    out_path = os.path.join(DATA_DIR, "extracted_signals_v1.csv")
    extracted_df[["lead_id", "sentiment", "urgency", "objection", "product_interest"]].to_csv(
        out_path, index=False
    )
    print(f"Saved extracted signals to {out_path}")

    # -----------------------------------------------------------------
    # Validation: compare extracted signals against the hidden tags
    # (the ground truth used to generate the notes in the first place)
    # -----------------------------------------------------------------
    hidden_path = os.path.join(DATA_DIR, "hidden_note_tags.csv")
    if os.path.exists(hidden_path):
        hidden_df = pd.read_csv(hidden_path)
        merged = extracted_df.merge(hidden_df, on="lead_id", suffixes=("_extracted", "_true"))

        print("\n--- Extraction accuracy vs hidden ground truth ---")
        field_map = {
            "sentiment": "note_sentiment",
            "urgency": "note_urgency",
            "objection": "note_objection",
            "product_interest": "note_product_interest",
        }
        for extracted_field, true_field in field_map.items():
            accuracy = (merged[extracted_field] == merged[true_field]).mean()
            print(f"{extracted_field:20s}: {accuracy:.1%}")
    else:
        print("\nNo hidden_note_tags.csv found, skipping accuracy validation.")


if __name__ == "__main__":
    main()