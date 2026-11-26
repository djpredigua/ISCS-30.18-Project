# 02_clean_ellna.py - Clean the ELLNA 2023-24 file

import pandas as pd
from common import DIVISION_MAP, LANG_MAP, LANG_PAIR_TYPE, TEXT_FIXES, save_sheet

RAW_FILE = "ellna_2023-24_selected.csv"
OTHER_FILE = "crla_2025-26_selected.csv"   # only used to find unmatched schools
TEXT_COLUMNS = ["region", "division", "language"]
SMALL_GROUP = 10   # fewer than 10 test takers = small group
SUSPECT_MPS = 1    # an MPS below 1 while the other scores are normal = likely entry error

def remove_duplicates(ellna):
    duplicates = ellna["school_id"].duplicated().sum()
    ellna = ellna.drop_duplicates(subset="school_id", keep="first")
    return ellna, duplicates

# Schools that are in this file but not in the other one (they stay in this file's table)
def count_unmatched(ellna):
    other_ids = pd.read_csv(OTHER_FILE, usecols=["school_id"])["school_id"]
    return (~ellna["school_id"].isin(other_ids)).sum()

# Blanks are reported, not filled: a blank score means the school did not take that test
def count_blanks(ellna):
    blanks = ellna.isna().sum()
    return blanks[blanks > 0]

def find_text_in_number_columns(ellna):
    number_columns = [column for column in ellna.columns if column not in TEXT_COLUMNS]
    return [column for column in number_columns if not pd.api.types.is_numeric_dtype(ellna[column])]


def trim_spaces(ellna):
    ellna = ellna.copy()
    changed = 0
    for column in TEXT_COLUMNS:
        trimmed = ellna[column].str.strip()
        changed = changed + (trimmed != ellna[column]).sum()
        ellna[column] = trimmed
    return ellna, changed


# Repair broken characters (TEXT_FIXES) and use the CRLA spelling of divisions (DIVISION_MAP)
def fix_text(ellna):
    ellna = ellna.copy()
    changed = 0
    for column in TEXT_COLUMNS:
        fixed = ellna[column]
        for broken, correct in TEXT_FIXES.items():
            fixed = fixed.str.replace(broken, correct, regex=False)
        if column == "division":
            fixed = fixed.replace(DIVISION_MAP)
        changed = changed + (fixed != ellna[column]).sum()
        ellna[column] = fixed
    return ellna, changed


def standardize_languages(ellna):

    # Raise error if not found in LANG_MAP
    unknown = sorted(set(ellna["language"]) - set(LANG_MAP))
    if unknown:
        raise ValueError(f"Add these language labels to LANG_MAP in common.py: {unknown}")

    ellna = ellna.copy()
    ellna["language_clean"] = ellna["language"].map(LANG_MAP)
    ellna["lang_pair_type"] = ellna["language_clean"].map(LANG_PAIR_TYPE)
    return ellna

# Flags: 1 = questionable, 0 = fine. Nothing is deleted, so results can be
# tested with and without these schools.
def add_flags(ellna):
    ellna = ellna.copy()

    # The school did not take the Mother Tongue test (score stays blank)
    ellna["mt_not_assessed"] = ellna["mother_tongue_mps"].isna().astype(int)

    # An MPS below 1 while the other scores are normal: likely an entry error.
    # Checked for every subject; MT gets its own flag because Indicator B depends on it.
    ellna["suspect_mt"] = (ellna["mother_tongue_mps"] < SUSPECT_MPS).astype(int)
    ellna["suspect_fil_eng"] = ((ellna["filipino_mps"] < SUSPECT_MPS)
                                | (ellna["english_mps"] < SUSPECT_MPS)).astype(int)
    ellna["suspect_num"] = (ellna["numeracy_mps"] < SUSPECT_MPS).astype(int)

    # Fewer than 10 test takers
    ellna["small_group"] = (ellna["n_test_takers"] < SMALL_GROUP).astype(int)
    return ellna

# Clean data and give summary of what happened
def step1_basics(ellna):
    report = []

    ellna, duplicates = remove_duplicates(ellna)
    report.append(("Duplicate school_id rows removed", duplicates))

    report.append(("Schools not in the CRLA file (kept)", count_unmatched(ellna)))

    report.append(("Blank cells", count_blanks(ellna).sum()))

    report.append(("Number columns that contain text", len(find_text_in_number_columns(ellna))))

    ellna, trimmed = trim_spaces(ellna)
    report.append(("Text cells with stray spaces trimmed", trimmed))

    ellna, fixed = fix_text(ellna)
    report.append(("Text cells fixed (broken characters, division spelling)", fixed))

    ellna = standardize_languages(ellna)
    report.append(("Schools whose language label was renamed or grouped",
                   (ellna["language"] != ellna["language_clean"]).sum()))

    return ellna, report

# Flag questionable schools and give summary
def step2_flags(ellna):
    ellna = add_flags(ellna)
    report = []
    for flag in ["mt_not_assessed", "suspect_mt", "suspect_fil_eng", "suspect_num", "small_group"]:
        report.append((f"Flag {flag} = 1", ellna[flag].sum()))
    return ellna, report


def main():
    # Import ELLNA file
    raw = pd.read_csv(RAW_FILE)
    print(f"Loaded {RAW_FILE}: {len(raw):,} schools, {raw.shape[1]} columns")

    ellna, report1 = step1_basics(raw)
    ellna, report2 = step2_flags(ellna)

    print("\nCleaning reports")
    for check, result in report1 + report2:
        print(f"  {check:<55}{int(result):>6,}")

    print("\nBlank cells by column (left blank on purpose):")
    print(count_blanks(ellna).to_string())

    print("\nText fixes (before -> after):")
    for column in TEXT_COLUMNS:
        changes = pd.DataFrame({"before": raw[column], "after": ellna[column]})
        changes = changes[changes["before"] != changes["after"]]
        if len(changes) > 0:
            print(changes.value_counts().reset_index(name="schools").to_string(index=False))

    print("\nLanguage labels (raw label -> standard name -> pair type):")
    labels = (ellna.groupby(["language", "language_clean", "lang_pair_type"]).size()
              .reset_index(name="schools").sort_values("schools", ascending=False))
    print(labels.to_string(index=False))

    # Save the cleaned table to its own sheet
    save_sheet(ellna, "1b Cleaned ELLNA")
    print(f"\nAfter Cleaning: {len(ellna):,} schools, {ellna.shape[1]} columns -> sheet '1b Cleaned ELLNA'")


if __name__ == "__main__":
    main()
