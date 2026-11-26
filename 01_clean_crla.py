# 01_clean_crla.py - Clean the CRLA 2025-26 file

import pandas as pd
from common import DIVISION_MAP, LANG_MAP, LANG_PAIR_TYPE, TEXT_FIXES, save_sheet

RAW_FILE = "crla_2025-26_selected.csv"
OTHER_FILE = "ellna_2023-24_selected.csv"   # only used to find unmatched schools
TEXT_COLUMNS = ["region", "division", "language"]

# Column names are a grade-language group + a reading level, e.g. "g1_mt_low_emerging"
GROUPS = ["g1_mt", "g2_mt", "g2_fil", "g3_mt", "g3_fil", "g3_eng"]
LEVELS = ["low_emerging", "high_emerging", "developing", "transitioning", "at_grade_level"]
SMALL_GROUP = 5   # fewer than 5 learners in a group = small group

def remove_duplicates(crla):
    duplicates = crla["school_id"].duplicated().sum()
    crla = crla.drop_duplicates(subset="school_id", keep="first")
    return crla, duplicates

# Schools that are in this file but not in the other one (they stay in this file's table)
def count_unmatched(crla):
    other_ids = pd.read_csv(OTHER_FILE, usecols=["school_id"])["school_id"]
    return (~crla["school_id"].isin(other_ids)).sum()

def count_blanks(crla):
    blanks = crla.isna().sum()
    return blanks[blanks > 0]

def find_text_in_number_columns(crla):
    number_columns = [column for column in crla.columns if column not in TEXT_COLUMNS]
    return [column for column in number_columns if not pd.api.types.is_numeric_dtype(crla[column])]


def trim_spaces(crla):
    crla = crla.copy()
    changed = 0
    for column in TEXT_COLUMNS:
        trimmed = crla[column].str.strip()
        changed = changed + (trimmed != crla[column]).sum()
        crla[column] = trimmed
    return crla, changed


# Repair broken characters (TEXT_FIXES) and use one spelling per division (DIVISION_MAP)
def fix_text(crla):
    crla = crla.copy()
    changed = 0
    for column in TEXT_COLUMNS:
        fixed = crla[column]
        for broken, correct in TEXT_FIXES.items():
            fixed = fixed.str.replace(broken, correct, regex=False)
        if column == "division":
            fixed = fixed.replace(DIVISION_MAP)
        changed = changed + (fixed != crla[column]).sum()
        crla[column] = fixed
    return crla, changed


def standardize_languages(crla):

    # Raise error if not found in LANG_Map
    unknown = sorted(set(crla["language"]) - set(LANG_MAP))
    if unknown:
        raise ValueError(f"Add these language labels to LANG_MAP in common.py: {unknown}")
    
    crla = crla.copy()
    crla["language_clean"] = crla["language"].map(LANG_MAP)
    crla["lang_pair_type"] = crla["language_clean"].map(LANG_PAIR_TYPE)
    return crla


# Number of schools with no learners assessed in each group. These counts stay 0 in the
# data; the mastery % for such a group must be left blank later, never 0%.
def count_empty_groups(crla):
    return {group: (crla[f"{group}_total"] == 0).sum() for group in GROUPS}


# True where two groups have the same count in all five reading levels (and some learners)
def same_counts(crla, group_a, group_b):
    same = crla[f"{group_a}_total"] > 0
    for level in LEVELS:
        same = same & (crla[f"{group_a}_{level}"] == crla[f"{group_b}_{level}"])
    return same


# Flags: 1 = questionable, 0 = fine. Nothing is deleted, so results can be
# tested with and without these schools.
def add_flags(crla):
    crla = crla.copy()

    # Grade 1 or Grade 2 Mother Tongue had no assessed learners
    crla["mt_not_assessed"] = ((crla["g1_mt_total"] == 0) | (crla["g2_mt_total"] == 0)).astype(int)

    # MT and Filipino counts identical in Grade 2 or Grade 3 (likely copy-entry).
    # Not flagged for Tagalog schools, where MT and Filipino are the same language.
    mt_equals_fil = same_counts(crla, "g2_mt", "g2_fil") | same_counts(crla, "g3_mt", "g3_fil")
    crla["mt_fil_identical"] = (mt_equals_fil & (crla["language_clean"] != "Tagalog")).astype(int)

    # Fewer than 5 learners in a group used by Indicator A
    crla["small_group"] = ((crla["g1_mt_total"] < SMALL_GROUP)
                           | (crla["g2_mt_total"] < SMALL_GROUP)
                           | (crla["g3_fil_total"] < SMALL_GROUP)
                           | (crla["g3_eng_total"] < SMALL_GROUP)).astype(int)
    return crla

# Clean data and give summary of what happened 
def step1_basics(crla):
    report = []

    crla, duplicates = remove_duplicates(crla)
    report.append(("Duplicate school_id rows removed", duplicates))

    report.append(("Schools not in the ELLNA file (kept)", count_unmatched(crla)))

    report.append(("Blank cells", count_blanks(crla).sum()))

    report.append(("Number columns that contain text", len(find_text_in_number_columns(crla))))

    crla, trimmed = trim_spaces(crla)
    report.append(("Text cells with stray spaces trimmed", trimmed))

    crla, fixed = fix_text(crla)
    report.append(("Text cells fixed (broken characters, division spelling)", fixed))

    crla = standardize_languages(crla)
    report.append(("Schools whose language label was renamed or grouped",
                   (crla["language"] != crla["language_clean"]).sum()))

    return crla, report

# Flag questionable schools and give summary
def step2_flags(crla):
    report = []

    for group, schools in count_empty_groups(crla).items():
        report.append((f"Schools with no learners assessed in {group}", schools))

    crla = add_flags(crla)
    for flag in ["mt_not_assessed", "mt_fil_identical", "small_group"]:
        report.append((f"Flag {flag} = 1", crla[flag].sum()))

    return crla, report


def main():
    # Import CRLA file 
    crla = pd.read_csv(RAW_FILE)
    print(f"Loaded {RAW_FILE}: {len(crla):,} schools, {crla.shape[1]} columns")

    crla, report1 = step1_basics(crla)
    crla, report2 = step2_flags(crla)

    print("\nCleaning reports")
    for check, result in report1 + report2:
        print(f"  {check:<55}{int(result):>6,}")

    print("\nLanguage labels (raw label -> standard name -> pair type):")
    labels = (crla.groupby(["language", "language_clean", "lang_pair_type"]).size()
              .reset_index(name="schools").sort_values("schools", ascending=False))
    print(labels.to_string(index=False))

    # Save the cleaned table to its own sheet
    save_sheet(crla, "1a Cleaned CRLA")
    print(f"\nAfter Cleaning: {len(crla):,} schools, {crla.shape[1]} columns -> sheet '1a Cleaned CRLA'")


if __name__ == "__main__":
    main()
