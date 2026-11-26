# 02_clean_ellna.py - Clean the ELLNA 2023-24 file

import pandas as pd
from common import DIVISION_MAP, LANG_MAP, LANG_PAIR_TYPE, TEXT_FIXES

RAW_FILE = "ellna_2023-24_selected.csv"
TEXT_COLUMNS = ["region", "division", "language"]

def remove_duplicates(ellna):
    duplicates = ellna["school_id"].duplicated().sum()
    ellna = ellna.drop_duplicates(subset="school_id", keep="first")
    return ellna, duplicates

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

# Clean data and give summary of what happened
def step1_basics(ellna):
    report = []

    ellna, duplicates = remove_duplicates(ellna)
    report.append(("Duplicate school_id rows removed", duplicates))

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


def main():
    # Import ELLNA file
    raw = pd.read_csv(RAW_FILE)
    print(f"Loaded {RAW_FILE}: {len(raw):,} schools, {raw.shape[1]} columns")

    ellna, report = step1_basics(raw)

    print("\nCleaning reports")
    for check, result in report:
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

    print(f"\nAfter Cleaning: {len(ellna):,} schools, {ellna.shape[1]} columns")


if __name__ == "__main__":
    main()
