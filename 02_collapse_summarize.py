# 02_collapse_summarize.py - Collapse and summarize the cleaned data
# Run after 01_clean_data.py (it reads sheets 1a and 1b from CachoPredigua.xlsx)
#
#   1. Collapse the columns: keep only what the indicators and the EDA use,
#      and turn the five CRLA reading levels into three bands
#   2. Language categories: schools per language pair type
#   3. Rows to groups: summaries by language pair type, region, language and division,
#      with the Indicator A / B correlation inside each group
#      (the division-level cross-check is in 03_eda.py, with the other correlations)
#   4. EDA summary: CRLA and ELLNA side by side by language and by region

import pandas as pd
from common import WORKBOOK, save_sheet

GROUPS = ["g1_mt", "g2_mt", "g2_fil", "g3_mt", "g3_fil", "g3_eng"]
GROUPINGS = [("Language pair type", "lang_pair_type"), ("Region", "region"),
             ("Language", "language_clean"), ("Division", "division")]
MIN_SCHOOLS = 10   # a correlation is only reported with at least 10 schools


# ---------------------------------------------------------------------------
# 1. Collapse the columns
#    Kept: school, language group, test takers, flags (so the EDA can filter),
#          and three reading bands + total for each grade-language group
#    Dropped: the raw language label (language_clean replaces it), the 10
#          school-level totals and percentages (not used by the indicators) and
#          the five original reading levels, now three bands:
#      Emerging                 = Low Emerging + High Emerging
#      Developing/Transitioning = Developing + Transitioning
#      At Grade Level           = unchanged
# ---------------------------------------------------------------------------
ID_COLUMNS = ["school_id", "region", "division", "language_clean", "lang_pair_type", "n_test_takers"]
CRLA_FLAGS = ["mt_not_assessed", "mt_fil_identical", "small_group"]
ELLNA_FLAGS = ["mt_not_assessed", "suspect_mt", "suspect_fil_eng", "suspect_num", "small_group"]
ELLNA_SCORES = ["mother_tongue_mps", "filipino_mps", "english_mps", "numeracy_mps"]

def collapse_bands(crla):
    collapsed = crla[ID_COLUMNS + CRLA_FLAGS].copy()
    for group in GROUPS:
        collapsed[f"{group}_emerging"] = crla[f"{group}_low_emerging"] + crla[f"{group}_high_emerging"]
        collapsed[f"{group}_dev_trans"] = crla[f"{group}_developing"] + crla[f"{group}_transitioning"]
        collapsed[f"{group}_at_grade_level"] = crla[f"{group}_at_grade_level"]
        collapsed[f"{group}_total"] = crla[f"{group}_total"]
    return collapsed


# ELLNA: drop the raw language label and overall_mps (the indicators compare the subject scores)
def collapse_ellna(ellna):
    return ellna[ID_COLUMNS + ELLNA_FLAGS + ELLNA_SCORES].copy()


# Share of all learners in each band, for all schools and per language pair type
def band_shares(collapsed):
    rows = []
    for sample in ["All schools", "Distinct L1", "Same language"]:
        schools = collapsed if sample == "All schools" else collapsed[collapsed["lang_pair_type"] == sample]
        for group in GROUPS:
            learners = schools[f"{group}_total"].sum()
            rows.append({
                "schools": sample,
                "group": group,
                "learners": learners,
                "% Emerging": round(100 * schools[f"{group}_emerging"].sum() / learners, 1),
                "% Developing/Transitioning": round(100 * schools[f"{group}_dev_trans"].sum() / learners, 1),
                "% At Grade Level": round(100 * schools[f"{group}_at_grade_level"].sum() / learners, 1),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Language categories: schools per language pair type in each file
# ---------------------------------------------------------------------------

def pair_type_counts(crla, ellna):
    counts = pd.DataFrame({"CRLA schools": crla["lang_pair_type"].value_counts(),
                           "ELLNA schools": ellna["lang_pair_type"].value_counts()})
    return counts.fillna(0).astype(int).reset_index(names="language pair type")


# ---------------------------------------------------------------------------
# 3. Rows to groups
# ---------------------------------------------------------------------------

# 100 x part / whole, blank when whole is 0 (nobody assessed - never 0%)
def percent(part, whole):
    return (100 * part / whole).where(whole > 0)

# School mastery % for Indicator A: early MT (Grade 1 + 2 pooled), Grade 3 Filipino and English
def add_mastery(crla):
    crla = crla.copy()
    crla["early_mt_pct"] = percent(crla["g1_mt_at_grade_level"] + crla["g2_mt_at_grade_level"],
                                   crla["g1_mt_total"] + crla["g2_mt_total"])
    crla["g3_fil_pct"] = percent(crla["g3_fil_at_grade_level"], crla["g3_fil_total"])
    crla["g3_eng_pct"] = percent(crla["g3_eng_at_grade_level"], crla["g3_eng_total"])
    return crla

# Pearson r in the main sample (small groups left out), only with 10+ schools
def correlation(schools, x, y):
    main = schools[schools["small_group"] == 0][[x, y]].dropna()
    if len(main) < MIN_SCHOOLS:
        return None
    return round(main[x].corr(main[y]), 3)

# Mean, median and SD that stay blank when a group has no scores
def mean(values):
    values = values.dropna()
    return round(values.mean(), 1) if len(values) > 0 else None

def median(values):
    values = values.dropna()
    return round(values.median(), 1) if len(values) > 0 else None

def sd(values):
    values = values.dropna()
    return round(values.std(), 1) if len(values) > 1 else None

# Mean where bigger schools count more (weighted by n_test_takers)
def weighted_mean(schools, column):
    has_score = schools[column].notna()
    weights = schools.loc[has_score, "n_test_takers"]
    if weights.sum() == 0:
        return None
    return round((schools.loc[has_score, column] * weights).sum() / weights.sum(), 1)


# One summary row for a group of CRLA schools
def crla_summary(schools):
    row = {"schools": len(schools),
           "learners": schools["n_test_takers"].sum(),
           "small_group flags": schools["small_group"].sum()}
    early_at_grade = (schools["g1_mt_at_grade_level"] + schools["g2_mt_at_grade_level"]).sum()
    early_total = (schools["g1_mt_total"] + schools["g2_mt_total"]).sum()
    row["early MT pooled %"] = round(100 * early_at_grade / early_total, 1)
    row["early MT mean %"] = mean(schools["early_mt_pct"])
    row["early MT median %"] = median(schools["early_mt_pct"])
    for group, name in [("g3_fil", "G3 Fil"), ("g3_eng", "G3 Eng")]:
        row[f"{name} pooled %"] = round(100 * schools[f"{group}_at_grade_level"].sum()
                                        / schools[f"{group}_total"].sum(), 1)
        row[f"{name} mean %"] = mean(schools[f"{group}_pct"])
        row[f"{name} median %"] = median(schools[f"{group}_pct"])
    row["r early MT vs G3 Fil"] = correlation(schools, "early_mt_pct", "g3_fil_pct")
    row["r early MT vs G3 Eng"] = correlation(schools, "early_mt_pct", "g3_eng_pct")
    return row


# One summary row for a group of ELLNA schools
def ellna_summary(schools):
    row = {"schools": len(schools),
           "test takers": schools["n_test_takers"].sum(),
           "schools with MT score": schools["mother_tongue_mps"].notna().sum(),
           "small_group flags": schools["small_group"].sum()}
    for column, name in [("mother_tongue_mps", "MT"), ("filipino_mps", "Fil"), ("english_mps", "Eng")]:
        row[f"{name} weighted mean"] = weighted_mean(schools, column)
        row[f"{name} mean"] = mean(schools[column])
        row[f"{name} median"] = median(schools[column])
        row[f"{name} SD"] = sd(schools[column])
    row["r MT vs Fil"] = correlation(schools, "mother_tongue_mps", "filipino_mps")
    row["r MT vs Eng"] = correlation(schools, "mother_tongue_mps", "english_mps")
    return row


# Summary rows for every language pair type, region, language and division, plus a total row
def summarize(data, summary_function):
    rows = []
    for label, column in GROUPINGS:
        for value, schools in data.groupby(column):
            rows.append({"grouped by": label, "group": value, **summary_function(schools)})
    rows.append({"grouped by": "All schools", "group": "All schools", **summary_function(data)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. EDA summary: CRLA and ELLNA side by side for each language and region
#    (both files use the same language names, so each language is one row)
# ---------------------------------------------------------------------------

def eda_summary(crla_table, ellna_table):
    crla_columns = {"schools": "CRLA schools", "early MT pooled %": "CRLA early MT %",
                    "G3 Fil pooled %": "CRLA G3 Fil %", "G3 Eng pooled %": "CRLA G3 Eng %",
                    "r early MT vs G3 Fil": "CRLA r MT-Fil", "r early MT vs G3 Eng": "CRLA r MT-Eng"}
    ellna_columns = {"schools": "ELLNA schools", "MT weighted mean": "ELLNA MT MPS",
                     "Fil weighted mean": "ELLNA Fil MPS", "Eng weighted mean": "ELLNA Eng MPS",
                     "r MT vs Fil": "ELLNA r MT-Fil", "r MT vs Eng": "ELLNA r MT-Eng"}
    keep = ["Language", "Region", "All schools"]
    crla_part = crla_table[crla_table["grouped by"].isin(keep)][["grouped by", "group"] + list(crla_columns)]
    ellna_part = ellna_table[ellna_table["grouped by"].isin(keep)][["grouped by", "group"] + list(ellna_columns)]
    both = crla_part.rename(columns=crla_columns).merge(
        ellna_part.rename(columns=ellna_columns), on=["grouped by", "group"], how="outer")
    # a language found in only one file leaves the other file's columns blank
    both[["CRLA schools", "ELLNA schools"]] = both[["CRLA schools", "ELLNA schools"]].astype("Int64")
    order = {"Language": 0, "Region": 1, "All schools": 2}
    return both.sort_values(["grouped by", "group"], key=lambda c: c.map(order) if c.name == "grouped by" else c)


def main():
    # Import the cleaned tables
    crla = pd.read_excel(WORKBOOK, sheet_name="1a Cleaned CRLA")
    ellna = pd.read_excel(WORKBOOK, sheet_name="1b Cleaned ELLNA")

    # 1. Collapse the columns
    collapsed = collapse_bands(crla)
    collapsed_ellna = collapse_ellna(ellna)
    shares = band_shares(collapsed)

    # 2. Language categories
    pair_types = pair_type_counts(crla, ellna)

    # 3. Rows to groups
    crla_table = summarize(add_mastery(crla), crla_summary)
    ellna_table = summarize(ellna, ellna_summary)

    # 4. EDA summary
    eda = eda_summary(crla_table, ellna_table)

    print(f"Collapsed CRLA: {crla.shape[1]} -> {collapsed.shape[1]} columns; "
          f"collapsed ELLNA: {ellna.shape[1]} -> {collapsed_ellna.shape[1]} columns\n")
    print("Learners in each band (all schools):")
    print(shares[shares["schools"] == "All schools"].to_string(index=False))
    print("\nSchools per language pair type:")
    print(pair_types.to_string(index=False))
    print("\nCRLA by language pair type:")
    print(crla_table[crla_table["grouped by"].isin(["Language pair type", "All schools"])]
          [["group", "schools", "early MT pooled %", "G3 Fil pooled %", "G3 Eng pooled %",
            "r early MT vs G3 Fil", "r early MT vs G3 Eng"]].to_string(index=False))
    print("\nELLNA by language pair type:")
    print(ellna_table[ellna_table["grouped by"].isin(["Language pair type", "All schools"])]
          [["group", "schools", "MT weighted mean", "Fil weighted mean", "Eng weighted mean",
            "r MT vs Fil", "r MT vs Eng"]].to_string(index=False))
    print("\nEDA summary (by language):")
    print(eda[eda["grouped by"] == "Language"].drop(columns="grouped by").to_string(index=False))

    # Save each table to its own sheet
    # Band shares and pair-type counts are only printed above (small tables);
    # the pair-type counts are also the "Language pair type" rows of the summaries
    save_sheet(collapsed, "2a Collapsed CRLA")
    save_sheet(collapsed_ellna, "2b Collapsed ELLNA")
    save_sheet(crla_table, "2c Summary CRLA")
    save_sheet(ellna_table, "2d Summary ELLNA")
    save_sheet(eda, "2e EDA summary")
    print("\nSaved sheets 2a to 2e to CachoPredigua.xlsx")


if __name__ == "__main__":
    main()
