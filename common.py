"""
common.py - maps shared by the CRLA and ELLNA cleaning scripts,
so both files are cleaned with exactly the same labels.
"""
import os

import pandas as pd

# Language labels as they appear in the raw files -> one standard name.
# The two files spell some languages differently (e.g. "Ilocano" in CRLA, "Ilokano/Iloko" in ELLNA).
# Languages with 7 or fewer schools are grouped as "Other MT" (too few schools to analyze alone).
LANG_MAP = {
    # spelled the same in both files
    "Tagalog": "Tagalog",
    "Waray": "Waray",
    "Kapampangan": "Kapampangan",
    "English/Filipino": "English/Filipino",
    # spelled differently in the two files
    "Ilocano": "Ilokano",                                      # CRLA
    "Ilokano/Iloko": "Ilokano",                                # ELLNA
    "Sinugbuanong Binisaya": "Sinugbuanong Binisaya",          # CRLA
    "Sinugbuanong Binisaya/Cebuano": "Sinugbuanong Binisaya",  # ELLNA
    "Pangasinan": "Pangasinan",                                # CRLA
    "Pangasinan/Pangasinense": "Pangasinan",                   # ELLNA
    # rare languages (7 or fewer schools)
    "Sambal": "Other MT",
    "Other": "Other MT",
    "Surigaonon": "Other MT",
    "Southern Sorsoganon": "Other MT",
    "Chavacano": "Other MT",
    "Bahasa Sug": "Other MT",
    "Yakan": "Other MT",
    "Ilokano/Iloko-Pangasinense": "Other MT",
}

# Standard language -> language pair type (how the mother tongue relates to Filipino)
LANG_PAIR_TYPE = {
    "Ilokano": "Distinct L1",                 # mother tongue is a different language from Filipino
    "Sinugbuanong Binisaya": "Distinct L1",
    "Pangasinan": "Distinct L1",
    "Waray": "Distinct L1",
    "Kapampangan": "Distinct L1",
    "Tagalog": "Same language",               # mother tongue and Filipino are the same language
    "English/Filipino": "No MT",              # no mother tongue test (ELLNA only)
    "Other MT": "Other",                      # the rare languages
}

# Division names spelled differently in ELLNA -> the CRLA spelling (for the ELLNA step)
DIVISION_MAP = {
    "Pangasinan I (Lingayen)": "Pangasinan I, Lingayen",
    "Pangasinan II (Binalonan)": "Pangasinan II, Binalonan",
    "San Jose del Monte City": "City of San Jose Del Monte",
    "Samar (Western)": "Samar (Western Samar)",
}

# Broken characters (UTF-8 text that was read the wrong way) -> the correct text (for the ELLNA step)
TEXT_FIXES = {
    "MuÃ±oz": "Muñoz",
}

# The workbook that holds the cleaned tables (one sheet per dataset)
WORKBOOK = "CachoPredigua.xlsx"


# Save a table to its own sheet. If the sheet is already there it is replaced;
# the other sheets in the workbook are left alone.
def save_sheet(table, sheet_name):
    if os.path.exists(WORKBOOK):
        writer = pd.ExcelWriter(WORKBOOK, engine="openpyxl", mode="a", if_sheet_exists="replace")
    else:
        writer = pd.ExcelWriter(WORKBOOK, engine="openpyxl")
    with writer:
        table.to_excel(writer, sheet_name=sheet_name, index=False)
