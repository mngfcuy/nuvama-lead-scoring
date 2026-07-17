import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no GUI backend, just saves files, lighter on resources
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.max_columns", None)
sns.set_theme(style="whitegrid")

PLOTS_DIR = "notebooks/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

df = pd.read_csv("data/leads_v1.csv")
print("shape:", df.shape)
print(df.head(3))

# ---------------------------------------------------------------
# 1. Basic structure and dtypes
# ---------------------------------------------------------------
print("\n--- df.info() ---")
df.info()

print("\n--- df.describe(include='all').T ---")
print(df.describe(include="all").T)

# ---------------------------------------------------------------
# 2. Class balance
# ---------------------------------------------------------------
print("\n--- class balance ---")
print(df["converted_within_30d"].value_counts())
print(df["converted_within_30d"].value_counts(normalize=True).round(3))

plt.figure(figsize=(5, 4))
sns.countplot(data=df, x="converted_within_30d")
plt.title("Class balance: converted_within_30d")
plt.xlabel("Converted within 30 days")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/01_class_balance.png", dpi=100)
plt.close()

# ---------------------------------------------------------------
# 3. Numeric feature distributions
# ---------------------------------------------------------------
numeric_cols = [
    "age",
    "annual_income_lakhs",
    "investable_surplus_lakhs",
    "days_since_first_contact",
    "touchpoints_last_30d",
    
]

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, col in zip(axes.flatten(), numeric_cols):
    sns.histplot(df[col], kde=True, ax=ax)
    ax.set_title(col)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/02_numeric_distributions.png", dpi=100)
plt.close()

# ---------------------------------------------------------------
# 4. Categorical feature breakdowns
# ---------------------------------------------------------------
categorical_cols = [
    "city_tier",
    "occupation",
    "source",
    "last_meeting_outcome",
    "note_sentiment",
    "note_urgency",
    "note_objection",
    "note_product_interest",
]

for col in categorical_cols:
    print(f"\n--- {col} ---")
    print(df[col].value_counts(normalize=True).round(3))

# ---------------------------------------------------------------
# 5. Numeric feature correlations
# ---------------------------------------------------------------
corr_cols = numeric_cols + ["conversion_probability", "converted_within_30d"]
corr_matrix = df[corr_cols].corr().round(2)
print("\n--- correlation matrix ---")
print(corr_matrix)

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0, vmin=-1, vmax=1)
plt.title("Correlation matrix: numeric features + outcome")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/03_correlation_heatmap.png", dpi=100)
plt.close()

# ---------------------------------------------------------------
# 6. Conversion rate by key categorical drivers
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

df.groupby("last_meeting_outcome")["converted_within_30d"].mean().sort_values().plot(
    kind="barh", ax=axes[0], title="Conversion rate by meeting outcome"
)
df.groupby("source")["converted_within_30d"].mean().sort_values().plot(
    kind="barh", ax=axes[1], title="Conversion rate by source"
)
df.groupby("note_sentiment")["converted_within_30d"].mean().sort_values().plot(
    kind="barh", ax=axes[2], title="Conversion rate by note sentiment"
)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/04_conversion_by_driver.png", dpi=100)
plt.close()

print(f"\nAll plots saved to {PLOTS_DIR}/")
print("Done. Open the PNGs in Finder or drag them into VS Code to view.")

# ---------------------------------------------------------------
# 7. Written observations
# ---------------------------------------------------------------
# Fill these in after looking at the printed output and PNGs above.
# Rough prompts to get you started, base the real numbers on what you see:
#
# 1. Class balance: what fraction of leads converted? Is it imbalanced enough
#    that you'll need to think about it during modeling (e.g. PR-AUC over
#    accuracy, maybe class weighting)?
# 2. Numeric distributions: any skew worth noting (e.g. investable_surplus_lakhs
#    likely right-skewed, a few high-surplus leads)? Any variable that looks
#    suspiciously uniform or oddly bucketed?
# 3. Correlation matrix: which numeric feature correlates most with
#    conversion_probability? Does that match the feature importance hierarchy
#    you already know from building the label formula?
# 4. Conversion_probability vs converted_within_30d correlation: how much
#    does the coin flip step reduce this compared to a perfect correlation of 1.0?
#    Good one to explain to your mentor, it's the noise you deliberately built in.
# 5. Categorical breakdowns: any category with very few leads (e.g. a
#    city_tier or occupation with low counts) that might behave unstably
#    once you split into train/test?
# 1. Class balance is moderately imbalanced: 77.2% did not convert (386 leads),
#    22.8% did (114 leads). Not extreme, but accuracy alone would be misleading
#    (a model predicting "no" every time gets 77% accuracy for free). PR-AUC and
#    a properly chosen threshold matter more here than plain accuracy.
#
# 2. annual_income_lakhs and investable_surplus_lakhs are both right-skewed
#    (income: mean 41.6, median 33.1, max 281.5; surplus: mean 14.6, median 10.3,
#    max 103.9), a small number of high-net-worth leads pull the mean up well
#    above the median. They're also strongly correlated with each other (0.75),
#    which is expected since surplus is a function of income, worth remembering
#    for modeling since it's a multicollinearity signal for the baseline model.
#
# 3. None of the numeric structured fields correlate strongly with
#    conversion_probability on their own, the highest is investable_surplus_lakhs
#    at just 0.10, followed by touchpoints_last_30d at 0.09. This matches the
#    label formula's design: last_meeting_outcome, source, and note_sentiment
#    (all categorical) are the dominant drivers, not the numeric demographic
#    fields. A baseline model using only numeric fields will likely underperform
#    one that properly encodes these categoricals.
#
# 4. conversion_probability correlates with the actual converted_within_30d
#    outcome at 0.70, not 1.0. That 0.30 gap is the noise deliberately introduced
#    by the coin flip step, even leads with a high latent conversion probability
#    don't always convert, and some low-probability leads still do. This is the
#    realistic noise ceiling for any model built on this data, no model should
#    be expected to perfectly separate the classes.
#
# 5. A few categories are thin: existing-client-family is only 8.8% of leads
#    (44 leads), and note_product_interest = AIF is only 7.8% (39 leads).
#    These small categories are worth watching once the data gets split into
#    train/test, a random split could easily leave very few of these leads in
#    one side, making that category's effect noisy or unstable to estimate.