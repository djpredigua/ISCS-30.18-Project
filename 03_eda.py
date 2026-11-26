# 03_eda.py - Exploratory data analysis
# Run after 02_collapse_summarize.py (it reads sheets 2a and 2b from CachoPredigua.xlsx)
#
# Guiding questions
#   Indicator A (CRLA): do schools with higher Grade 1-2 Mother Tongue mastery also show
#                       higher Grade 3 Filipino and English mastery?
#   Indicator B (ELLNA): do schools with higher mother_tongue_mps also achieve higher
#                        filipino_mps and english_mps?
#
#   1. Check the top, bottom and sample sizes
#   2. Test the simple solution first: scatter plots + Pearson r
#   3. Challenge the findings: language pair types, weighting, within-grade pairs,
#      flagged records removed, and the numeracy comparison
#   4. Coarse cross-check: correlation of division averages (from the summaries)

import os

import matplotlib
matplotlib.use("Agg")   # save plots to files without opening a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from openpyxl import load_workbook
from openpyxl.drawing.image import Image

from common import WORKBOOK, save_sheet

FIGURES = "figures"
MIN_SCHOOLS = 10   # a correlation is only reported with at least 10 schools

# The pairs we correlate: (name, x column, y column)
CRLA_PAIRS = [("A: early MT vs G3 Filipino", "early_mt_pct", "g3_fil_pct"),
              ("A: early MT vs G3 English", "early_mt_pct", "g3_eng_pct")]
ELLNA_PAIRS = [("B: MT vs Filipino MPS", "mother_tongue_mps", "filipino_mps"),
               ("B: MT vs English MPS", "mother_tongue_mps", "english_mps")]
WITHIN_GRADE_PAIRS = [("G2 MT vs G2 Filipino", "g2_mt_pct", "g2_fil_pct"),
                      ("G3 MT vs G3 Filipino", "g3_mt_pct", "g3_fil_pct"),
                      ("G3 MT vs G3 English", "g3_mt_pct", "g3_eng_pct")]


# 100 x part / whole, blank when whole is 0 (nobody assessed - never 0%)
def percent(part, whole):
    return (100 * part / whole).where(whole > 0)

# Mastery % = learners at grade level / learners assessed, for every CRLA group
def add_mastery(crla):
    crla = crla.copy()
    for group in ["g1_mt", "g2_mt", "g2_fil", "g3_mt", "g3_fil", "g3_eng"]:
        crla[f"{group}_pct"] = percent(crla[f"{group}_at_grade_level"], crla[f"{group}_total"])
    crla["early_mt_pct"] = percent(crla["g1_mt_at_grade_level"] + crla["g2_mt_at_grade_level"],
                                   crla["g1_mt_total"] + crla["g2_mt_total"])
    return crla


# Pearson r for the schools that have both values. Returns (schools, r);
# r is blank when there are fewer than 10 schools.
def correlation(schools, x, y):
    both = schools[[x, y]].dropna()
    if len(both) < MIN_SCHOOLS:
        return len(both), None
    return len(both), round(both[x].corr(both[y]), 3)

# The same r, but bigger schools count more (weighted by n_test_takers)
def weighted_correlation(schools, x, y):
    both = schools[[x, y, "n_test_takers"]].dropna()
    if len(both) < MIN_SCHOOLS:
        return len(both), None
    w = both["n_test_takers"]
    x_dev = both[x] - (both[x] * w).sum() / w.sum()
    y_dev = both[y] - (both[y] * w).sum() / w.sum()
    r = (w * x_dev * y_dev).sum() / ((w * x_dev ** 2).sum() * (w * y_dev ** 2).sum()) ** 0.5
    return len(both), round(r, 3)


# Adds one row per pair to the results table
def add_rows(results, step, dataset, sample, schools, pairs, weighted=False):
    for pair, x, y in pairs:
        if weighted:
            n, r = weighted_correlation(schools, x, y)
        else:
            n, r = correlation(schools, x, y)
        results.append({"step": step, "dataset": dataset, "pair": pair, "sample": sample,
                        "schools": n, "r": r})


# Correlation of division averages (divisions with 10+ schools), next to the school-level r
def division_rows(results):
    crla = pd.read_excel(WORKBOOK, sheet_name="2c Summary CRLA")
    ellna = pd.read_excel(WORKBOOK, sheet_name="2d Summary ELLNA")
    crla_div = crla[(crla["grouped by"] == "Division") & (crla["schools"] >= MIN_SCHOOLS)]
    ellna_div = ellna[(ellna["grouped by"] == "Division") & (ellna["schools with MT score"] >= MIN_SCHOOLS)]
    pairs = [("CRLA", "A: early MT vs G3 Filipino", crla_div, "early MT pooled %", "G3 Fil pooled %"),
             ("CRLA", "A: early MT vs G3 English", crla_div, "early MT pooled %", "G3 Eng pooled %"),
             ("ELLNA", "B: MT vs Filipino MPS", ellna_div, "MT weighted mean", "Fil weighted mean"),
             ("ELLNA", "B: MT vs English MPS", ellna_div, "MT weighted mean", "Eng weighted mean")]
    for dataset, pair, divisions, x, y in pairs:
        results.append({"step": "4. Division averages", "dataset": dataset, "pair": pair,
                        "sample": "Divisions with 10+ schools (one row per division)",
                        "schools": len(divisions), "r": round(divisions[x].corr(divisions[y]), 3)})


# ---------------------------------------------------------------------------
# 1. Check the top, bottom and sample sizes
# ---------------------------------------------------------------------------

# Top and bottom 10 schools (ties broken by size, smallest first, to show if tiny schools sit at the extremes)
def top_and_bottom(crla, ellna):
    crla_columns = ["school_id", "division", "language_clean", "n_test_takers", "g3_fil_total", "g3_fil_pct",
                    "early_mt_pct", "small_group", "mt_fil_identical"]
    ellna_columns = ["school_id", "division", "language_clean", "n_test_takers", "mother_tongue_mps",
                     "filipino_mps", "english_mps", "small_group", "suspect_mt"]
    crla_sorted = crla.dropna(subset=["g3_fil_pct"]).sort_values(["g3_fil_pct", "g3_fil_total"],
                                                                 ascending=[False, True])
    ellna_sorted = ellna.dropna(subset=["mother_tongue_mps"]).sort_values(["mother_tongue_mps", "n_test_takers"],
                                                                         ascending=[False, True])
    crla_table = pd.concat([crla_sorted.head(10)[crla_columns].assign(list="top 10"),
                            crla_sorted.tail(10)[crla_columns].assign(list="bottom 10")])
    ellna_table = pd.concat([ellna_sorted.head(10)[ellna_columns].assign(list="top 10"),
                             ellna_sorted.tail(10)[ellna_columns].assign(list="bottom 10")])
    # put the "list" column first
    crla_table = crla_table[["list"] + crla_columns]
    ellna_table = ellna_table[["list"] + ellna_columns]
    return crla_table, ellna_table


# How much a school's G3 Filipino mastery swings depending on how many learners were tested
def size_check(crla):
    crla = crla.dropna(subset=["g3_fil_pct"]).copy()
    crla["G3 Filipino learners"] = pd.cut(crla["g3_fil_total"], [0, 4, 9, 19, 100000],
                                          labels=["1-4", "5-9", "10-19", "20 or more"])
    rows = []
    for size, schools in crla.groupby("G3 Filipino learners", observed=True):
        rows.append({"G3 Filipino learners": size,
                     "schools": len(schools),
                     "mean %": round(schools["g3_fil_pct"].mean(), 1),
                     "SD": round(schools["g3_fil_pct"].std(), 1),
                     "% of schools at exactly 0% or 100%": round(100 * schools["g3_fil_pct"].isin([0, 100]).mean(), 1)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Scatter plots (main sample: small groups left out)
# ---------------------------------------------------------------------------

def scatter(ax, schools, x, y, title, xlabel, ylabel, color, trend_line=False):
    both = schools[[x, y]].dropna()
    ax.scatter(both[x], both[y], s=5, alpha=0.3, color=color)

    # Trend line: the straight line that best fits the dots (least squares)
    if trend_line:
        slope, intercept = np.polyfit(both[x], both[y], 1)
        line_x = np.array([0, 100])
        ax.plot(line_x, intercept + slope * line_x, color="black", linewidth=2,
                label=f"trend: y = {intercept:.1f} + {slope:.2f}x")
        ax.legend(loc="upper left", fontsize=8)

    n, r = correlation(schools, x, y)
    ax.set_title(f"{title}\nr = {r}, {n:,} schools", fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

def make_plots(crla_main, ellna_main):
    os.makedirs(FIGURES, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    scatter(axes[0], crla_main, "early_mt_pct", "g3_fil_pct", "CRLA: early MT vs Grade 3 Filipino",
            "Early MT mastery % (G1-G2)", "G3 Filipino mastery %", "#257de0")
    scatter(axes[1], crla_main, "early_mt_pct", "g3_eng_pct", "CRLA: early MT vs Grade 3 English",
            "Early MT mastery % (G1-G2)", "G3 English mastery %", "#257de0")
    fig.tight_layout()
    fig.savefig(f"{FIGURES}/indicator_a_crla.png", dpi=110)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    scatter(axes[0], ellna_main, "mother_tongue_mps", "filipino_mps", "ELLNA: MT vs Filipino",
            "Mother tongue MPS", "Filipino MPS", "#f47230", trend_line=True)
    scatter(axes[1], ellna_main, "mother_tongue_mps", "english_mps", "ELLNA: MT vs English",
            "Mother tongue MPS", "English MPS", "#f47230", trend_line=True)
    scatter(axes[2], ellna_main, "mother_tongue_mps", "numeracy_mps", "ELLNA: MT vs Numeracy (comparison)",
            "Mother tongue MPS", "Numeracy MPS", "#00b894", trend_line=True)
    fig.tight_layout()
    fig.savefig(f"{FIGURES}/indicator_b_ellna.png", dpi=110)
    plt.close(fig)

# Put the two plot images on one sheet of the workbook
def save_plots_sheet():
    book = load_workbook(WORKBOOK)
    if "3b Scatter plots" in book.sheetnames:
        del book["3b Scatter plots"]
    sheet = book.create_sheet("3b Scatter plots")
    sheet["A1"] = "Main sample (small groups left out). Each dot is one school."
    sheet.add_image(Image(f"{FIGURES}/indicator_a_crla.png"), "A3")
    sheet.add_image(Image(f"{FIGURES}/indicator_b_ellna.png"), "A30")
    book.save(WORKBOOK)


def main():
    crla = add_mastery(pd.read_excel(WORKBOOK, sheet_name="2a Collapsed CRLA"))
    ellna = pd.read_excel(WORKBOOK, sheet_name="2b Collapsed ELLNA")
    crla_main = crla[crla["small_group"] == 0]      # main sample: small groups left out
    ellna_main = ellna[ellna["small_group"] == 0]
    results = []

    # 1. Top, bottom and sample sizes
    extremes_crla, extremes_ellna = top_and_bottom(crla, ellna)
    sizes = size_check(crla)

    # 2. The simple solution: Pearson r across all valid schools, then the main sample
    add_rows(results, "2. Simple r", "CRLA", "All valid schools", crla, CRLA_PAIRS)
    add_rows(results, "2. Simple r", "ELLNA", "All valid schools", ellna, ELLNA_PAIRS)
    add_rows(results, "2. Simple r", "CRLA", "Main sample (no small groups)", crla_main, CRLA_PAIRS)
    add_rows(results, "2. Simple r", "ELLNA", "Main sample (no small groups)", ellna_main, ELLNA_PAIRS)
    make_plots(crla_main, ellna_main)

    # 3a. Within each language pair type, unweighted and weighted by n_test_takers
    for pair_type in ["Distinct L1", "Same language", "Other"]:
        schools = crla_main[crla_main["lang_pair_type"] == pair_type]
        add_rows(results, "3a. By pair type", "CRLA", pair_type, schools, CRLA_PAIRS)
        add_rows(results, "3a. By pair type", "CRLA", f"{pair_type}, weighted", schools, CRLA_PAIRS, weighted=True)
    for pair_type in ["Distinct L1"]:   # only Distinct L1 schools have an ELLNA MT score
        schools = ellna_main[ellna_main["lang_pair_type"] == pair_type]
        add_rows(results, "3a. By pair type", "ELLNA", pair_type, schools, ELLNA_PAIRS)
        add_rows(results, "3a. By pair type", "ELLNA", f"{pair_type}, weighted", schools, ELLNA_PAIRS, weighted=True)
    add_rows(results, "3a. By pair type", "CRLA", "All, weighted", crla_main, CRLA_PAIRS, weighted=True)
    add_rows(results, "3a. By pair type", "ELLNA", "All, weighted", ellna_main, ELLNA_PAIRS, weighted=True)

    # 3b. Within-grade CRLA pairs (the same learners take both tests)
    add_rows(results, "3b. Within grade", "CRLA", "Main sample", crla_main, WITHIN_GRADE_PAIRS)
    distinct = crla_main[crla_main["lang_pair_type"] == "Distinct L1"]
    add_rows(results, "3b. Within grade", "CRLA", "Distinct L1", distinct, WITHIN_GRADE_PAIRS)
    add_rows(results, "3b. Within grade", "CRLA", "Distinct L1, no mt_fil_identical",
             distinct[distinct["mt_fil_identical"] == 0], WITHIN_GRADE_PAIRS)
    tagalog = crla_main[crla_main["lang_pair_type"] == "Same language"]
    add_rows(results, "3b. Within grade", "CRLA", "Same language (Tagalog)", tagalog, WITHIN_GRADE_PAIRS)

    # 3c. Rerun without flagged records
    add_rows(results, "3c. Without flags", "CRLA", "Without mt_fil_identical",
             crla_main[crla_main["mt_fil_identical"] == 0], CRLA_PAIRS)
    add_rows(results, "3c. Without flags", "CRLA", "Without any flag",
             crla_main[(crla_main["mt_fil_identical"] == 0) & (crla_main["mt_not_assessed"] == 0)], CRLA_PAIRS)
    add_rows(results, "3c. Without flags", "ELLNA", "Without suspect_mt",
             ellna_main[ellna_main["suspect_mt"] == 0], ELLNA_PAIRS)
    no_flags = ellna_main[(ellna_main["suspect_mt"] == 0) & (ellna_main["suspect_fil_eng"] == 0)
                          & (ellna_main["suspect_num"] == 0)]
    add_rows(results, "3c. Without flags", "ELLNA", "Without any flag", no_flags, ELLNA_PAIRS)

    # 3d. Numeracy comparison: same schools, language pairs vs a non-language pair
    numeracy_pairs = ELLNA_PAIRS + [("Comparison: MT vs Numeracy MPS", "mother_tongue_mps", "numeracy_mps")]
    same_schools = ellna_main.dropna(subset=["mother_tongue_mps", "filipino_mps", "english_mps", "numeracy_mps"])
    add_rows(results, "3d. Numeracy check", "ELLNA", "Main sample", same_schools, numeracy_pairs)
    add_rows(results, "3d. Numeracy check", "ELLNA", "Without any flag",
             same_schools[(same_schools["suspect_mt"] == 0) & (same_schools["suspect_fil_eng"] == 0)
                          & (same_schools["suspect_num"] == 0)], numeracy_pairs)

    # 4. Division averages (the "schools" column counts divisions here)
    division_rows(results)

    correlations = pd.DataFrame(results)

    print("CRLA top and bottom 10 (G3 Filipino mastery):")
    print(extremes_crla.to_string(index=False))
    print("\nELLNA top and bottom 10 (MT MPS):")
    print(extremes_ellna.to_string(index=False))
    print("\nSample size check (Grade 3 Filipino mastery):")
    print(sizes.to_string(index=False))
    print("\nCorrelations:")
    print(correlations.to_string(index=False))

    # Top/bottom lists and the sample size check are only printed above
    save_sheet(correlations, "3a Correlations")
    save_plots_sheet()
    print(f"\nSaved sheets 3a Correlations and 3b Scatter plots to {WORKBOOK} and plots to the {FIGURES} folder")


if __name__ == "__main__":
    main()
