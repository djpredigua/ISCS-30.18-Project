"""
Tests for Step 1 of 01_clean_crla.py.

Run from the project folder:   python test_clean_crla.py

The real CRLA file has no duplicates, blanks or stray spaces, so most tests use a
tiny made-up table that has them, to prove each check would catch them.
"""
import importlib

import pandas as pd

from common import LANG_MAP, LANG_PAIR_TYPE

# A file name that starts with a number cannot be imported the normal way
clean = importlib.import_module("01_clean_crla")


def made_up_table():
    """Four schools with problems planted on purpose:
    school 100002 appears twice, "Region I" has stray spaces, and one count is blank."""
    return pd.DataFrame({
        "school_id": [100001, 100002, 100002, 100003],
        "region": ["CAR", " Region I ", " Region I ", "Region III"],
        "division": ["Abra", "Ilocos Norte", "Ilocos Norte", "Bulacan"],
        "language": ["Ilocano", "Sambal", "Sambal", "Tagalog"],
        "n_test_takers": [15, 20, 20, None],
    })


def test_duplicates_are_removed():
    table, removed = clean.remove_duplicates(made_up_table())
    assert removed == 1
    assert table["school_id"].tolist() == [100001, 100002, 100003]


def test_blanks_are_counted():
    assert clean.count_blanks(made_up_table()).to_dict() == {"n_test_takers": 1}


def test_text_in_a_number_column_is_found():
    table = made_up_table()
    table["n_test_takers"] = ["15", "20", "N/A", "12"]
    assert clean.find_text_in_number_columns(table) == ["n_test_takers"]


def test_stray_spaces_are_trimmed():
    table, changed = clean.trim_spaces(made_up_table())
    assert changed == 2
    assert table["region"].tolist() == ["CAR", "Region I", "Region I", "Region III"]


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


def test_every_standard_language_has_a_pair_type():
    assert set(LANG_MAP.values()) <= set(LANG_PAIR_TYPE)


def test_real_crla_file():
    crla = pd.read_csv("crla_2025-26_selected.csv")
    cleaned, report = clean.step1_basics(crla)
    assert len(cleaned) == 6121                        # no school lost
    assert cleaned["school_id"].is_unique
    assert cleaned["language_clean"].notna().all()     # every school has a standard language
    assert cleaned["lang_pair_type"].notna().all()
    assert sorted(cleaned["language_clean"].unique()) == [
        "Ilokano", "Kapampangan", "Other MT", "Pangasinan", "Sinugbuanong Binisaya", "Tagalog", "Waray"]


if __name__ == "__main__":
    tests = [test_duplicates_are_removed, test_blanks_are_counted, test_text_in_a_number_column_is_found,
             test_stray_spaces_are_trimmed, test_languages_are_standardized,
             test_an_unknown_language_stops_the_script, test_every_standard_language_has_a_pair_type,
             test_real_crla_file]
    for test in tests:
        test()
        print("passed:", test.__name__)
    print(f"\nAll {len(tests)} tests passed")
