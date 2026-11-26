# 04_features.py - Feature engineering and selection
# Run after 02_collapse_summarize.py (it reads sheets 2a and 2b from CachoPredigua.xlsx)
#
#   1. Feature selection: keep identifiers, test-taker totals, the CRLA at-grade-level
#      counts and totals, and the ELLNA subject scores
#   2. Mastery % features (Indicator A): 100 x at grade level / total, on the same
#      0-100 scale as the ELLNA scores
#   3. Proficiency gaps: L2/L3 minus L1
#   4. Flags (0/1) and the language pair type (category) for filtering and subgroups

import pandas as pd
from common import WORKBOOK, save_sheet

IDENTIFIERS = ["school_id", "region", "division", "language_clean", "lang_pair_type", "n_test_takers"]
CRLA_FLAGS = ["mt_not_assessed", "mt_fil_identical", "small_group"]
ELLNA_FLAGS = ["mt_not_assessed", "suspect_mt", "suspect_fil_eng", "suspect_num", "small_group"]
CRLA_GROUPS = ["g1_mt", "g2_mt", "g2_fil", "g3_mt", "g3_fil", "g3_eng"]
ELLNA_SCORES = ["mother_tongue_mps", "filipino_mps", "english_mps", "numeracy_mps"]


# 100 x part / whole, blank when whole is 0 (nobody assessed - never 0%)
def percent(part, whole):
    return (100 * part / whole).where(whole > 0)


def crla_features(crla):
    # 1. Feature selection: identifiers, flags, and the at-grade-level count + total of each group
    count_columns = []
    for group in CRLA_GROUPS:
        count_columns = count_columns + [f"{group}_at_grade_level", f"{group}_total"]
    features = crla[IDENTIFIERS + CRLA_FLAGS + count_columns].copy()

    # 2. Mastery % for each grade-language group
    for group in CRLA_GROUPS:
        features[f"{group}_pct"] = percent(crla[f"{group}_at_grade_level"], crla[f"{group}_total"])

    # Early MT mastery: Grade 1 and Grade 2 Mother Tongue learners pooled (the L1 baseline)
    features["early_mt_pct"] = percent(crla["g1_mt_at_grade_level"] + crla["g2_mt_at_grade_level"],
                                       crla["g1_mt_total"] + crla["g2_mt_total"])

    # 3. Proficiency gaps (blank when either side is blank)
    features["gap_g3_fil_early_mt"] = features["g3_fil_pct"] - features["early_mt_pct"]
    features["gap_g3_eng_early_mt"] = features["g3_eng_pct"] - features["early_mt_pct"]

    # 4. Language pair type as a category
    features["lang_pair_type"] = features["lang_pair_type"].astype("category")
    return features


def ellna_features(ellna):
    # 1. Feature selection: identifiers, flags and the four subject scores
    features = ellna[IDENTIFIERS + ELLNA_FLAGS + ELLNA_SCORES].copy()

    # 3. Proficiency gaps (blank when the MT score is blank)
    features["gap_fil_mt"] = features["filipino_mps"] - features["mother_tongue_mps"]
    features["gap_eng_mt"] = features["english_mps"] - features["mother_tongue_mps"]

    # 4. Language pair type as a category
    features["lang_pair_type"] = features["lang_pair_type"].astype("category")
    return features


# Quick look at the new features: how many schools have a value, and the mean
def describe(features, columns):
    rows = []
    for column in columns:
        values = features[column].dropna()
        rows.append({"feature": column, "schools with a value": len(values),
                     "mean": round(values.mean(), 1), "median": round(values.median(), 1)})
    return pd.DataFrame(rows)


def main():
    crla = pd.read_excel(WORKBOOK, sheet_name="2a Collapsed CRLA")
    ellna = pd.read_excel(WORKBOOK, sheet_name="2b Collapsed ELLNA")

    crla_table = crla_features(crla)
    ellna_table = ellna_features(ellna)

    crla_new = [f"{group}_pct" for group in CRLA_GROUPS] + ["early_mt_pct", "gap_g3_fil_early_mt", "gap_g3_eng_early_mt"]
    ellna_new = ["gap_fil_mt", "gap_eng_mt"]
    print(f"CRLA features: {crla_table.shape[0]:,} schools, {crla_table.shape[1]} columns")
    print(describe(crla_table, crla_new).to_string(index=False))
    print(f"\nELLNA features: {ellna_table.shape[0]:,} schools, {ellna_table.shape[1]} columns")
    print(describe(ellna_table, ellna_new).to_string(index=False))

    save_sheet(crla_table, "4a Features CRLA")
    save_sheet(ellna_table, "4b Features ELLNA")
    print("\nSaved sheets 4a Features CRLA and 4b Features ELLNA to CachoPredigua.xlsx")


if __name__ == "__main__":
    main()
