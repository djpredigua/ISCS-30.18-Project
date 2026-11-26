# 01_clean_data.py - Clean the CRLA 2025-26 and ELLNA 2023-24 files

import pandas as pd
from common import DIVISION_MAP, LANG_MAP, LANG_PAIR_TYPE, TEXT_FIXES, save_sheet

CRLA_FILE = "crla_2025-26_selected.csv"
ELLNA_FILE = "ellna_2023-24_selected.csv"
TEXT_COLUMNS = ["region", "division", "language"]

# CRLA column names are a grade-language group + a reading level, e.g. "g1_mt_low_emerging"
GROUPS = ["g1_mt", "g2_mt", "g2_fil", "g3_mt", "g3_fil", "g3_eng"]
LEVELS = ["low_emerging", "high_emerging", "developing", "transitioning", "at_grade_level"]

CRLA_SMALL_GROUP = 5    # fewer than 5 learners in a CRLA group = small group
ELLNA_SMALL_GROUP = 10  # fewer than 10 ELLNA test takers = small group
SUSPECT_MPS = 1         # an MPS below 1 while the other scores are normal = likely entry error



# Basic checks done for both files 

def remove_duplicates(data):
    duplicates = data["school_id"].duplicated().sum()
    data = data.drop_duplicates(subset="school_id", keep="first")
    return data, duplicates

# Schools that are in this file but not in the other one (they stay in this file's table)
def count_unmatched(data, other):
    return (~data["school_id"].isin(other["school_id"])).sum()

# Blanks are reported, not filled: a blank score means the school did not take that test
def count_blanks(data):
    blanks = data.isna().sum()
    return blanks[blanks > 0]

def find_text_in_number_columns(data):
    number_columns = [column for column in data.columns if column not in TEXT_COLUMNS]
    return [column for column in number_columns if not pd.api.types.is_numeric_dtype(data[column])]

def trim_spaces(data):
    data = data.copy()
    changed = 0
    for column in TEXT_COLUMNS:
        trimmed = data[column].str.strip()
        changed = changed + (trimmed != data[column]).sum()
        data[column] = trimmed
    return data, changed

# Repair broken characters (TEXT_FIXES) and use one spelling per division (DIVISION_MAP)
def fix_text(data):
    data = data.copy()
    changed = 0
    for column in TEXT_COLUMNS:
        fixed = data[column]
        for broken, correct in TEXT_FIXES.items():
            fixed = fixed.str.replace(broken, correct, regex=False)
        if column == "division":
            fixed = fixed.replace(DIVISION_MAP)
        changed = changed + (fixed != data[column]).sum()
        data[column] = fixed
    return data, changed


def standardize_languages(data):

    # Raise error if not found in LANG_MAP
    unknown = sorted(set(data["language"]) - set(LANG_MAP))
    if unknown:
        raise ValueError(f"Add these language labels to LANG_MAP in common.py: {unknown}")

    data = data.copy()
    data["language_clean"] = data["language"].map(LANG_MAP)
    data["lang_pair_type"] = data["language_clean"].map(LANG_PAIR_TYPE)
    return data

# Clean data and give summary of what happened
def step1_basics(data, other, other_name):
    report = []

    data, duplicates = remove_duplicates(data)
    report.append(("Duplicate school_id rows removed", duplicates))

    report.append((f"Schools not in the {other_name} file (kept)", count_unmatched(data, other)))

    report.append(("Blank cells", count_blanks(data).sum()))

    report.append(("Number columns that contain text", len(find_text_in_number_columns(data))))

    data, trimmed = trim_spaces(data)
    report.append(("Text cells with stray spaces trimmed", trimmed))

    data, fixed = fix_text(data)
    report.append(("Text cells fixed (broken characters, division spelling)", fixed))

    data = standardize_languages(data)
    report.append(("Schools whose language label was renamed or grouped",
                   (data["language"] != data["language_clean"]).sum()))

    return data, report


# Flags: 1 = questionable, 0 = fine. DO NOT REMOVE, used for checking data integirty later on

# True where two CRLA groups have the same count in all five reading levels (and some learners)
def same_counts(crla, group_a, group_b):
    same = crla[f"{group_a}_total"] > 0
    for level in LEVELS:
        same = same & (crla[f"{group_a}_{level}"] == crla[f"{group_b}_{level}"])
    return same


def crla_flags(crla):
    crla = crla.copy()
    report = []

    # Groups with no assessed learners: counts stay 0, their mastery % must be left blank later
    for group in GROUPS:
        report.append((f"Schools with no learners assessed in {group}", (crla[f"{group}_total"] == 0).sum()))

    # Grade 1 or Grade 2 Mother Tongue had no assessed learners
    crla["mt_not_assessed"] = ((crla["g1_mt_total"] == 0) | (crla["g2_mt_total"] == 0)).astype(int)

    # MT and Filipino counts identical in Grade 2 or Grade 3 (likely copy-entry).
    # Not flagged for Tagalog schools, where MT and Filipino are the same language.
    mt_equals_fil = same_counts(crla, "g2_mt", "g2_fil") | same_counts(crla, "g3_mt", "g3_fil")
    crla["mt_fil_identical"] = (mt_equals_fil & (crla["language_clean"] != "Tagalog")).astype(int)

    # Fewer than 5 learners in a group used by Indicator A
    crla["small_group"] = ((crla["g1_mt_total"] < CRLA_SMALL_GROUP)
                           | (crla["g2_mt_total"] < CRLA_SMALL_GROUP)
                           | (crla["g3_fil_total"] < CRLA_SMALL_GROUP)
                           | (crla["g3_eng_total"] < CRLA_SMALL_GROUP)).astype(int)

    for flag in ["mt_not_assessed", "mt_fil_identical", "small_group"]:
        report.append((f"Flag {flag} = 1", crla[flag].sum()))
    return crla, report


def ellna_flags(ellna):
    ellna = ellna.copy()
    report = []

    # The school did not take the Mother Tongue test (score stays blank)
    ellna["mt_not_assessed"] = ellna["mother_tongue_mps"].isna().astype(int)

    # An MPS below 1 while the other scores are normal: likely an entry error.
    # Checked for every subject; MT gets its own flag because Indicator B depends on it.
    ellna["suspect_mt"] = (ellna["mother_tongue_mps"] < SUSPECT_MPS).astype(int)
    ellna["suspect_fil_eng"] = ((ellna["filipino_mps"] < SUSPECT_MPS)
                                | (ellna["english_mps"] < SUSPECT_MPS)).astype(int)
    ellna["suspect_num"] = (ellna["numeracy_mps"] < SUSPECT_MPS).astype(int)

    # Fewer than 10 test takers
    ellna["small_group"] = (ellna["n_test_takers"] < ELLNA_SMALL_GROUP).astype(int)

    for flag in ["mt_not_assessed", "suspect_mt", "suspect_fil_eng", "suspect_num", "small_group"]:
        report.append((f"Flag {flag} = 1", ellna[flag].sum()))
    return ellna, report


# Print report

def print_report(name, raw, data, report):
    print(f"\n===== {name}: {len(raw):,} schools, {raw.shape[1]} columns loaded =====")
    for check, result in report:
        print(f"  {check:<55}{int(result):>6,}")

    blanks = count_blanks(data)
    if len(blanks) > 0:
        print("\nBlank cells by column (left blank on purpose):")
        print(blanks.to_string())

    print("\nText fixes (before -> after):")
    fixes_found = False
    for column in TEXT_COLUMNS:
        changes = pd.DataFrame({"before": raw[column], "after": data[column]})
        changes = changes[changes["before"] != changes["after"]]
        if len(changes) > 0:
            print(changes.value_counts().reset_index(name="schools").to_string(index=False))
            fixes_found = True
    if not fixes_found:
        print("  none")

    print("\nLanguage labels (raw label -> standard name -> pair type):")
    labels = (data.groupby(["language", "language_clean", "lang_pair_type"]).size()
              .reset_index(name="schools").sort_values("schools", ascending=False))
    print(labels.to_string(index=False))


def main():
    # Import both files
    crla_raw = pd.read_csv(CRLA_FILE)
    ellna_raw = pd.read_csv(ELLNA_FILE)

    # CRLA
    crla, report1 = step1_basics(crla_raw, ellna_raw, "ELLNA")
    crla, report2 = crla_flags(crla)
    print_report("CRLA 2025-26", crla_raw, crla, report1 + report2)

    # ELLNA
    ellna, report1 = step1_basics(ellna_raw, crla_raw, "CRLA")
    ellna, report2 = ellna_flags(ellna)
    print_report("ELLNA 2023-24", ellna_raw, ellna, report1 + report2)

    # Save each cleaned table to its own sheet
    save_sheet(crla, "1a Cleaned CRLA")
    save_sheet(ellna, "1b Cleaned ELLNA")
    print(f"\nSaved: '1a Cleaned CRLA' ({len(crla):,} x {crla.shape[1]}) and "
          f"'1b Cleaned ELLNA' ({len(ellna):,} x {ellna.shape[1]}) to CachoPredigua.xlsx")


if __name__ == "__main__":
    main()
