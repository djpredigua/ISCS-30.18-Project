import os
import numpy as np
import pandas as pd


LANG_MAP = {
    # Standardized single canonical names for multi-variant labels 
    "Ilocano": "Ilokano",
    "Ilokano/Iloko": "Ilokano",
    "Sinugbuanong Binisaya": "Sinugbuanong Binisaya",
    "Sinugbuanong Binisaya/Cebuano": "Sinugbuanong Binisaya",
    "Pangasinan": "Pangasinan",
    "Pangasinan/Pangasinense": "Pangasinan",
    "Tagalog": "Tagalog",
    "Waray": "Waray",
    "Kapampangan": "Kapampangan",
    "English/Filipino": "English/Filipino",
    # Rare test languages (<= 7 schools in either dataset) -> Other MT
    "Sambal": "Other MT",
    "Other": "Other MT",
    "Surigaonon": "Other MT",
    "Southern Sorsoganon": "Other MT",
    "Chavacano": "Other MT",
    "Bahasa Sug": "Other MT",
    "Yakan": "Other MT",
    "Ilokano/Iloko-Pangasinense": "Other MT",
}

MPS_COLS = [
    "mother_tongue_mps",
    "filipino_mps",
    "english_mps",
    "numeracy_mps",
    "overall_mps",
]

def classify_lang_pair(lang_clean: str) -> str:
    if lang_clean == "Tagalog":
        return "Same language"
    if lang_clean == "English/Filipino":
        return "No MT"
    if lang_clean == "Other MT":
        return "Other"
    return "Distinct L1"


def main() -> None:
    os.makedirs("outputs", exist_ok=True)

    # 1. Load raw ELLNA 2023-24 dataset
    ellna = pd.read_csv("ellna_2023-24_selected.csv")
    print(f"Loaded ELLNA raw shape: {ellna.shape}")

    # 2. Deduplicate on school_id (keep first occurrence)
    dup_count = int(ellna["school_id"].duplicated().sum())
    print(f"Duplicate school_id rows found: {dup_count}")
    ellna = ellna.drop_duplicates(subset=["school_id"], keep="first").copy()

    # 3. Clean string columns (trim whitespace & fix UTF-8/Latin-1 artifacts)
    for col in ["region", "division", "language"]:
        ellna[col] = (
            ellna[col]
            .astype(str)
            .str.strip()
            .str.replace("MuÃ±oz", "Muñoz", regex=False)
        )
        assert not ellna[col].str.contains("Ã", regex=False).any(), f"Encoding artifact remains in {col}"

    # 4. Validate test-taker counts and 0-100 MPS bounds
    assert (ellna["n_test_takers"] > 0).all(), "Non-positive n_test_takers found in ELLNA!"
    for col in MPS_COLS:
        valid = ellna[col].dropna()
        assert ((valid >= 0.0) & (valid <= 100.0)).all(), f"Out-of-range MPS values in {col}"

    # 5. Harmonize language labels & assign language pair category
    ellna["language_clean"] = ellna["language"].map(LANG_MAP)
    assert ellna["language_clean"].isna().sum() == 0, "Unmapped language found in ELLNA!"
    ellna["lang_pair_type"] = ellna["language_clean"].apply(classify_lang_pair)

    # 6. Create 0/1 data quality flags (retain all rows & keep missing mother_tongue_mps blank)
    # mt_not_assessed: 1 if mother_tongue_mps is missing (unassessed in Mother Tongue)
    ellna["mt_not_assessed"] = ellna["mother_tongue_mps"].isna().astype(int)

    # suspect_mt: 1 if mother_tongue_mps is recorded but < 1.0 (extreme low/zero coding outlier)
    ellna["suspect_mt"] = (
        ellna["mother_tongue_mps"].notna() & (ellna["mother_tongue_mps"] < 1.0)
    ).astype(int)

    # small_group: 1 if school has < 10 ELLNA test takers
    ellna["small_group"] = (ellna["n_test_takers"] < 10).astype(int)

    # 7. Export Worksheet 2: Cleaned ELLNA
    cleaned_cols = [
        "school_id", "region", "division", "language", "language_clean", "lang_pair_type",
        "mt_not_assessed", "suspect_mt", "small_group", "n_test_takers",
        "mother_tongue_mps", "filipino_mps", "english_mps", "numeracy_mps", "overall_mps",
    ]
    ellna[cleaned_cols].to_csv("outputs/ellna_cleaned.csv", index=False)
    
if __name__ == "__main__":
    main()