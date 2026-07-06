"""
generate_notes.py

Generates rm_notes free text for each lead, PLUS a set of "hidden tags"
(note_sentiment, note_urgency, note_objection, note_product_interest).

Important: the hidden tags are our own ground truth for validating the LLM/
keyword extractor later (Phase 7 / Day 5). They are saved to a SEPARATE file
(hidden_note_tags.csv) and must NOT be fed into the model as features -- only
the raw rm_notes text is a real feature. The tags exist so we can check:
"did our extractor correctly recover sentiment/urgency/objection from the text?"

Run standalone for testing: python generate_notes.py
"""

import numpy as np
import pandas as pd

SEED = 43  # different seed from structured fields, kept independent
rng = np.random.default_rng(SEED)


# ---- Template banks ----------------------------------------------------
# Each list holds several phrasings so notes don't feel copy-pasted.

POSITIVE_TEMPLATES = [
    "Client sounded genuinely interested and asked good follow-up questions.",
    "Very engaged on the call, seems ready to move forward soon.",
    "Client mentioned they want to invest before month end.",
    "Positive tone throughout, comfortable discussing larger ticket sizes.",
    "Client compared us favorably to their current advisor.",
    "Enthusiastic about the proposal, asked for next steps immediately.",
]

NEUTRAL_TEMPLATES = [
    "Client listened but didn't commit either way.",
    "Standard intro call, no strong signal yet.",
    "Client said they'd think it over and get back to us.",
    "Polite conversation, nothing concrete decided.",
    "Client is still exploring options, non-committal for now.",
]

NEGATIVE_TEMPLATES = [
    "Client seemed hesitant and raised concerns.",
    "Not very responsive on the call, seemed distracted.",
    "Client pushed back on several points, unclear if this will progress.",
    "Conversation felt one-sided, client gave short answers.",
    "Client expressed some doubt about moving forward right now.",
]

URGENCY_HIGH_ADDONS = [
    "Wants to close this quickly.",
    "Asked to be contacted again within the week.",
    "Said timing is important for them right now.",
]

URGENCY_LOW_ADDONS = [
    "No particular rush on their end.",
    "Said there's no urgency, can revisit in a few months.",
    "Mentioned they're in no hurry to decide.",
]

OBJECTION_TEMPLATES = {
    "Pricing": [
        "Raised concerns about fees compared to other options.",
        "Felt the cost structure was too high for the ticket size.",
    ],
    "Trust": [
        "Wanted more proof of past performance before committing.",
        "Seemed unsure about the firm's track record.",
    ],
    "Timing": [
        "Said this isn't the right time given current market conditions.",
        "Wants to wait until after a personal financial event before deciding.",
    ],
    "Competitor": [
        "Mentioned they're also evaluating another wealth firm.",
        "Said a competing advisor already made a similar pitch.",
    ],
}

PRODUCT_TEMPLATES = {
    "MF": ["Asked about mutual fund options.", "Interested in SIP-based mutual fund plans."],
    "PMS": ["Curious about portfolio management services for a larger allocation.", "Asked what PMS minimums look like."],
    "AIF": ["Asked about alternative investment funds.", "Wanted to understand AIF risk profile."],
    "SIP": ["Interested in starting a systematic investment plan.", "Asked about monthly SIP amounts."],
    "Unclear": ["Didn't specify a particular product.", "General interest, no specific product named yet."],
}


def sample_sentiment_from_outcome(last_meeting_outcome):
    """
    Sentiment isn't independent of last_meeting_outcome -- it's correlated,
    but not perfectly (an RM's note can still diverge a little from the
    literal 'Positive/Neutral/Objection' outcome tag).
    """
    if last_meeting_outcome == "Positive":
        probs = {"Positive": 0.75, "Neutral": 0.20, "Negative": 0.05}
    elif last_meeting_outcome == "Objection":
        probs = {"Positive": 0.05, "Neutral": 0.25, "Negative": 0.70}
    elif last_meeting_outcome == "No-show":
        probs = {"Positive": 0.05, "Neutral": 0.35, "Negative": 0.60}
    elif last_meeting_outcome == "Neutral":
        probs = {"Positive": 0.25, "Neutral": 0.55, "Negative": 0.20}
    else:  # None-yet
        probs = {"Positive": 0.20, "Neutral": 0.60, "Negative": 0.20}

    options = list(probs.keys())
    weights = list(probs.values())
    return rng.choice(options, p=weights)


def sample_urgency(touchpoints):
    """More touchpoints loosely raises the odds of high urgency language."""
    if touchpoints >= 7:
        probs = [0.55, 0.30, 0.15]
    elif touchpoints >= 3:
        probs = [0.30, 0.45, 0.25]
    else:
        probs = [0.15, 0.40, 0.45]
    return rng.choice(["High", "Medium", "Low"], p=probs)


def sample_product_interest(surplus):
    """Higher surplus -> more plausible to be pitched PMS/AIF."""
    if surplus >= 30:
        probs = [0.15, 0.35, 0.25, 0.10, 0.15]
    elif surplus >= 10:
        probs = [0.30, 0.15, 0.10, 0.30, 0.15]
    else:
        probs = [0.35, 0.05, 0.05, 0.35, 0.20]
    return rng.choice(["MF", "PMS", "AIF", "SIP", "Unclear"], p=probs)


def build_note(sentiment, urgency, objection, product_interest):
    sentences = []

    # 1 sentence from sentiment bank
    bank = {"Positive": POSITIVE_TEMPLATES, "Neutral": NEUTRAL_TEMPLATES, "Negative": NEGATIVE_TEMPLATES}[sentiment]
    sentences.append(rng.choice(bank))

    # 1 sentence on urgency (skip sometimes for variety)
    if urgency == "High":
        sentences.append(rng.choice(URGENCY_HIGH_ADDONS))
    elif urgency == "Low" and rng.random() < 0.7:
        sentences.append(rng.choice(URGENCY_LOW_ADDONS))

    # objection sentence, only if sentiment is Negative
    if sentiment == "Negative" and objection != "None":
        sentences.append(rng.choice(OBJECTION_TEMPLATES[objection]))

    # product interest sentence
    sentences.append(rng.choice(PRODUCT_TEMPLATES[product_interest]))

    # shuffle order slightly, cap at 2-5 sentences per brief
    rng.shuffle(sentences)
    return " ".join(sentences)


def generate_notes_for_leads(df):
    """
    df must already have: last_meeting_outcome, touchpoints_last_30d,
    investable_surplus_lakhs columns.
    Returns (rm_notes list, hidden_tags DataFrame)
    """
    rm_notes = []
    hidden_rows = []

    for _, row in df.iterrows():
        sentiment = sample_sentiment_from_outcome(row["last_meeting_outcome"])
        urgency = sample_urgency(row["touchpoints_last_30d"])
        product_interest = sample_product_interest(row["investable_surplus_lakhs"])

        if sentiment == "Negative":
            objection = rng.choice(["Pricing", "Trust", "Timing", "Competitor"])
        else:
            objection = "None"

        note = build_note(sentiment, urgency, objection, product_interest)
        rm_notes.append(note)
        hidden_rows.append({
            "lead_id": row["lead_id"],
            "note_sentiment": sentiment,
            "note_urgency": urgency,
            "note_objection": objection,
            "note_product_interest": product_interest,
        })

    hidden_tags = pd.DataFrame(hidden_rows)
    return rm_notes, hidden_tags


if __name__ == "__main__":
    # standalone test using the structured file from Phase 2
    df = pd.read_csv("../data/leads_v1_structured.csv")
    notes, hidden_tags = generate_notes_for_leads(df)
    df["rm_notes"] = notes

    print(df[["lead_id", "last_meeting_outcome", "touchpoints_last_30d", "rm_notes"]].head(8).to_string())
    print("\nHidden tag distribution:")
    print(hidden_tags["note_sentiment"].value_counts())

    hidden_tags.to_csv("../data/hidden_note_tags.csv", index=False)
    print("\nSaved hidden tags to data/hidden_note_tags.csv (NOT a model feature -- validation only)")
