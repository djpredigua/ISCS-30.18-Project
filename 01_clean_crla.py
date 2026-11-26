# 01_clean_crla.py - Clean the CRLA 2025-26 file

import pandas as pd
from common import LANG_MAP, LANG_PAIR_TYPE

RAW_FILE = "crla_2025-26_selected.csv"
TEXT_COLUMNS = ["region", "division", "language"]

def remove_duplicates(crla):
    duplicates = crla["school_id"].duplicated().sum()
    crla = crla.drop_duplicates(subset="school_id", keep="first")
    return crla, duplicates

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


def standardize_languages(crla):

    # Raise error if not found in LANG_Map
    unknown = sorted(set(crla["language"]) - set(LANG_MAP))
    if unknown:
        raise ValueError(f"Add these language labels to LANG_MAP in common.py: {unknown}")
    
    crla = crla.copy()
    crla["language_clean"] = crla["language"].map(LANG_MAP)
    crla["lang_pair_type"] = crla["language_clean"].map(LANG_PAIR_TYPE)
    return crla

# Clean data and give summary of what happened 
def step1_basics(crla):
    report = []

    crla, duplicates = remove_duplicates(crla)
    report.append(("Duplicate school_id rows removed", duplicates))

    report.append(("Blank cells", count_blanks(crla).sum()))

    report.append(("Number columns that contain text", len(find_text_in_number_columns(crla))))

    crla, trimmed = trim_spaces(crla)
    report.append(("Text cells with stray spaces trimmed", trimmed))

    crla = standardize_languages(crla)
    report.append(("Schools whose language label was renamed or grouped",
                   (crla["language"] != crla["language_clean"]).sum()))

    return crla, report


def main():
    # Import CRLA file 
    crla = pd.read_csv(RAW_FILE)
    print(f"Loaded {RAW_FILE}: {len(crla):,} schools, {crla.shape[1]} columns")

    crla, report = step1_basics(crla)

    print("\nCleaning reports")
    for check, result in report:
        print(f"  {check:<55}{int(result):>6,}")

    print("\nLanguage labels (raw label -> standard name -> pair type):")
    labels = (crla.groupby(["language", "language_clean", "lang_pair_type"]).size()
              .reset_index(name="schools").sort_values("schools", ascending=False))
    print(labels.to_string(index=False))

    print(f"\nAfter Cleaning: {len(crla):,} schools, {crla.shape[1]} columns")


if __name__ == "__main__":
    main()