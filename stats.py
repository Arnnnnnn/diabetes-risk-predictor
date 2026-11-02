"""Statistical tests on the diabetes dataset.

Run from the folder that contains diabetes_prediction_dataset.csv:
    python stats.py
"""
import numpy as np
import pandas as pd
from scipy import stats

def fp(p):
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


# ---- Load and clean (same steps as clean.py) ----
df = pd.read_csv("diabetes_prediction_dataset.csv")
df = df.drop_duplicates().reset_index(drop=True)
df["smoking_history"] = df["smoking_history"].map({
    "never": "never", "current": "current", "former": "former",
    "not current": "former", "ever": "former", "No Info": "unknown",
})
df = df[df["gender"] != "Other"].reset_index(drop=True)
print("Rows used:", len(df))
print()

# ---- Test 1: HbA1c, diabetic vs non-diabetic (Mann-Whitney U) ----
# Mann-Whitney does not assume the values are normally distributed.
pos = df.loc[df["diabetes"] == 1, "HbA1c_level"]
neg = df.loc[df["diabetes"] == 0, "HbA1c_level"]
u, p1 = stats.mannwhitneyu(pos, neg, alternative="two-sided")
# Effect size: probability that a random diabetic person has a higher HbA1c
# than a random non-diabetic person (1.0 = always higher, 0.5 = no difference).
prob_superiority = u / (len(pos) * len(neg))
print("TEST 1: HbA1c, diabetic vs non-diabetic (Mann-Whitney U)")
print(f"  Median HbA1c, diabetic:     {pos.median():.2f}")
print(f"  Median HbA1c, non-diabetic: {neg.median():.2f}")
print(f"  p-value: {fp(p1)}")
print(f"  Probability a diabetic person has higher HbA1c: {prob_superiority:.3f}")
print()

# ---- Test 2: Welch t-test on blood glucose, with Cohen's d ----
g_pos = df.loc[df["diabetes"] == 1, "blood_glucose_level"]
g_neg = df.loc[df["diabetes"] == 0, "blood_glucose_level"]
t, p2 = stats.ttest_ind(g_pos, g_neg, equal_var=False)
pooled_sd = np.sqrt((g_pos.var(ddof=1) + g_neg.var(ddof=1)) / 2)
cohens_d = (g_pos.mean() - g_neg.mean()) / pooled_sd
print("TEST 2: Blood glucose, diabetic vs non-diabetic (Welch t-test)")
print(f"  Mean glucose, diabetic:     {g_pos.mean():.1f}")
print(f"  Mean glucose, non-diabetic: {g_neg.mean():.1f}")
print(f"  p-value: {fp(p2)}")
print(f"  Cohen's d: {cohens_d:.2f}")
print()

# ---- Test 3: Chi-square, smoking history vs diabetes ----
table = pd.crosstab(df["smoking_history"], df["diabetes"])
chi2, p3, dof, _ = stats.chi2_contingency(table)
cramers_v = np.sqrt(chi2 / (table.values.sum() * (min(table.shape) - 1)))
rates = (table[1] / table.sum(axis=1) * 100).round(1)
print("TEST 3: Smoking history vs diabetes (chi-square test of independence)")
print("  Diabetes rate (%) by smoking history:")
print(rates.to_string())
print(f"  p-value: {fp(p3)}")
print(f"  Cramer's V: {cramers_v:.3f}")
print()

# ---- Test 4: One-way ANOVA, BMI across smoking groups ----
groups = [g["bmi"].values for _, g in df.groupby("smoking_history")]
f, p4 = stats.f_oneway(*groups)
grand = df["bmi"].mean()
ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
ss_total = ((df["bmi"] - grand) ** 2).sum()
eta_sq = ss_between / ss_total
print("TEST 4: BMI across smoking groups (one-way ANOVA)")
print("  Mean BMI by group:")
print(df.groupby("smoking_history")["bmi"].mean().round(2).to_string())
print(f"  F = {f:.1f}, p-value: {fp(p4)}")
print(f"  Eta squared: {eta_sq:.4f}")
print()

print("Note: with about 96,000 rows almost any difference gives a tiny p-value,")
print("so read the effect sizes (probability, Cohen's d, Cramer's V, eta squared).")
