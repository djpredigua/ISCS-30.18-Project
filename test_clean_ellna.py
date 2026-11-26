# Tests for 02_clean_ellna.py - run from the project folder:   python test_clean_ellna.py
# Most tests use a tiny made-up table with problems planted on purpose,
# to prove each check would catch them.

import importlib

import pandas as pd

# A file name that starts with a number cannot be imported the normal way
clean = importlib.import_module("02_clean_ellna")


def made_up_table():
    # school 100002 appears twice, " Region I " has stray spaces, one MT score is blank,
    # one division has a broken character and one uses the ELLNA spelling
    return pd.DataFrame({
        "school_id": [100001, 100002, 100002, 100003],
        "region": ["Region I", " Region I ", " Region I ", "Region III"],
        "division": ["Samar (Western)", "Ilocos Norte", "Ilocos Norte", "Science City of MuÃ±oz"],
        "language": ["Ilokano/Iloko", "Yakan", "Yakan", "Tagalog"],
        "mother_tongue_mps": [75.7, 60.1, 60.1, None],
    })


def test_duplicates_are_removed():
    table, removed = clean.remove_duplicates(made_up_table())
    assert removed == 1
    assert table["school_id"].tolist() == [100001, 100002, 100003]


def test_blanks_are_counted():
    assert clean.count_blanks(made_up_table()).to_dict() == {"mother_tongue_mps": 1}


def test_text_in_a_number_column_is_found():
    table = made_up_table()
    table["mother_tongue_mps"] = ["75.7", "60.1", "N/A", "50"]
    assert clean.find_text_in_number_columns(table) == ["mother_tongue_mps"]


def test_stray_spaces_are_trimmed():
    table, changed = clean.trim_spaces(made_up_table())
    assert changed == 2
    assert table["region"].tolist() == ["Region I", "Region I", "Region I", "Region III"]


def test_broken_characters_and_division_spellings_are_fixed():
    table, changed = clean.fix_text(made_up_table())
    assert changed == 2
    assert table["division"].tolist() == ["Samar (Western Samar)", "Ilocos Norte", "Ilocos Norte",
                                          "Science City of Muñoz"]


def test_languages_are_standardized():
    table = clean.standardize_languages(made_up_table())
    assert table["language_clean"].tolist() == ["Ilokano", "Other MT", "Other MT", "Tagalog"]
    assert table["lang_pair_type"].tolist() == ["Distinct L1", "Other", "Other", "Same language"]


def test_an_unknown_language_stops_the_script():
    table = made_up_table()
    table.loc[0, "language"] = "Spanish"
    try:
        clean.standardize_languages(table)
    except ValueError:
        return
    raise AssertionError("An unknown language label was not caught")


def test_real_ellna_file():
    ellna = pd.read_csv("ellna_2023-24_selected.csv")
    cleaned, report = clean.step1_basics(ellna)
    assert len(cleaned) == 5752                                  # no school lost
    assert cleaned["school_id"].is_unique
    assert cleaned["language_clean"].notna().all()
    assert not cleaned["division"].str.contains("Ã").any()       # no broken characters left
    assert not cleaned["division"].isin(clean.DIVISION_MAP.keys()).any()   # only CRLA spellings
    # blank MT scores are kept blank, and only in schools that did not take the MT test
    assert cleaned["mother_tongue_mps"].isna().sum() == 1925
    assert set(cleaned.loc[cleaned["mother_tongue_mps"].isna(), "language_clean"]) == {"Tagalog", "English/Filipino"}


def test_divisions_now_match_crla():
    crla = pd.read_csv("crla_2025-26_selected.csv")
    ellna, report = clean.step1_basics(pd.read_csv("ellna_2023-24_selected.csv"))
    assert set(ellna["division"]) == set(crla["division"])


if __name__ == "__main__":
    tests = [test_duplicates_are_removed, test_blanks_are_counted, test_text_in_a_number_column_is_found,
             test_stray_spaces_are_trimmed, test_broken_characters_and_division_spellings_are_fixed,
             test_languages_are_standardized, test_an_unknown_language_stops_the_script,
             test_real_ellna_file, test_divisions_now_match_crla]
    for test in tests:
        test()
        print("passed:", test.__name__)
    print(f"\nAll {len(tests)} tests passed")
