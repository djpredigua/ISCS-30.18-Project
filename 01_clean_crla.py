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

GRADE_LANG_GROUPS = ["g1_mt", "g2_mt", "g2_fil", "g3_mt", "g3_fil", "g3_eng"]
BAND_SUFFIXES = [
    "low_emerging",
    "high_emerging",
    "developing",
    "transitioning",
    "at_grade_level",
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

    # 1. Load raw CRLA 2025-26 dataset
    crla = pd.read_csv("crla_2025-26_selected.csv")
    print(f"Loaded CRLA raw shape: {crla.shape}")

    # 2. Deduplicate on school_id (keep first occurrence)
    dup_count = int(crla["school_id"].duplicated().sum())
    print(f"Duplicate school_id rows found: {dup_count}")
    crla = crla.drop_duplicates(subset=["school_id"], keep="first").copy()

    # 3. Clean string columns (trim whitespace & fix UTF-8/Latin-1 artifacts)
    for col in ["region", "division", "language"]:
        crla[col] = (
            crla[col]
            .astype(str)
            .str.strip()
            .str.replace("MuÃ±oz", "Muñoz", regex=False)
        )

    # 4. Validate learner count integrity across all 6 grade-language groups
    for prefix in GRADE_LANG_GROUPS:
        band_cols = [f"{prefix}_{b}" for b in BAND_SUFFIXES]
        assert (crla[band_cols] >= 0).all().all(), f"Negative counts found in {prefix}"
        band_sum = crla[band_cols].sum(axis=1)
        assert (band_sum == crla[f"{prefix}_total"]).all(), f"Band sum mismatch in {prefix}"

    # 5. Harmonize language labels & assign language pair category
    crla["language_clean"] = crla["language"].map(LANG_MAP)
    assert crla["language_clean"].isna().sum() == 0, "Unmapped language found in CRLA!"
    crla["lang_pair_type"] = crla["language_clean"].apply(classify_lang_pair)

    # 6. Create 0/1 data quality flags (retain all rows; exclude only in sensitivity checks)
    # mt_not_assessed: 1 if Grade 1 or Grade 2 Mother Tongue has 0 assessed learners
    crla["mt_not_assessed"] = (
        (crla["g1_mt_total"] == 0) | (crla["g2_mt_total"] == 0)
    ).astype(int)

    # mt_fil_identical: 1 if non-Tagalog school has identical MT and Filipino counts in Grade 2 or Grade 3
    g2_mt_cols = [f"g2_mt_{b}" for b in BAND_SUFFIXES] + ["g2_mt_total"]
    g2_fil_cols = [f"g2_fil_{b}" for b in BAND_SUFFIXES] + ["g2_fil_total"]
    g3_mt_cols = [f"g3_mt_{b}" for b in BAND_SUFFIXES] + ["g3_mt_total"]
    g3_fil_cols = [f"g3_fil_{b}" for b in BAND_SUFFIXES] + ["g3_fil_total"]

    g2_ident = (crla[g2_mt_cols].values == crla[g2_fil_cols].values).all(axis=1) & (crla["g2_mt_total"] > 0)
    g3_ident = (crla[g3_mt_cols].values == crla[g3_fil_cols].values).all(axis=1) & (crla["g3_mt_total"] > 0)
    crla["mt_fil_identical"] = (
        (crla["language_clean"] != "Tagalog") & (g2_ident | g3_ident)
    ).astype(int)

    # small_group: 1 if < 5 assessed learners in any core Indicator A grade-language group
    crla["small_group"] = (
        (crla["g1_mt_total"] < 5)
        | (crla["g2_mt_total"] < 5)
        | (crla["g3_fil_total"] < 5)
        | (crla["g3_eng_total"] < 5)
    ).astype(int)

    # 7. Export Worksheet 1: Cleaned CRLA
    cleaned_cols = [
        "school_id", "region", "division", "language", "language_clean", "lang_pair_type",
        "mt_not_assessed", "mt_fil_identical", "small_group", "n_test_takers",
        "school_low_emerging_total", "school_high_emerging_total", "school_developing_total",
        "school_transitioning_total", "school_at_grade_level_total",
        "school_low_emerging_percent", "school_high_emerging_percent", "school_developing_percent",
        "school_transitioning_percent", "school_at_grade_level_percent",
    ]
    for prefix in GRADE_LANG_GROUPS:
        cleaned_cols.extend([f"{prefix}_{b}" for b in BAND_SUFFIXES] + [f"{prefix}_total"])

    crla[cleaned_cols].to_csv("outputs/crla_cleaned.csv", index=False)

    # 8. Feature Engineering (0-100 mastery rates, proficiency gaps, collapsed 3-band counts)
    fe = crla.copy()

    # Mastery percentage features (0-100 scale; zero-denominator groups stay blank/NaN)
    for prefix in GRADE_LANG_GROUPS:
        fe[f"{prefix}_pct"] = np.where(
            fe[f"{prefix}_total"] > 0,
            100.0 * fe[f"{prefix}_at_grade_level"] / fe[f"{prefix}_total"],
            np.nan,
        )

    early_tot = fe["g1_mt_total"] + fe["g2_mt_total"]
    early_agl = fe["g1_mt_at_grade_level"] + fe["g2_mt_at_grade_level"]
    fe["early_mt_pct"] = np.where(early_tot > 0, 100.0 * early_agl / early_tot, np.nan)

    # Proficiency gap features (L2/L3 minus L1)
    fe["gap_g3_fil_early_mt"] = fe["g3_fil_pct"] - fe["early_mt_pct"]
    fe["gap_g3_eng_early_mt"] = fe["g3_eng_pct"] - fe["early_mt_pct"]
    fe["gap_g2_fil_mt"] = fe["g2_fil_pct"] - fe["g2_mt_pct"]
    fe["gap_g3_fil_mt"] = fe["g3_fil_pct"] - fe["g3_mt_pct"]
    fe["gap_g3_eng_mt"] = fe["g3_eng_pct"] - fe["g3_mt_pct"]

    # Collapsed 3-band counts across all 6 grade-language groups:
    # Band 1: Emerging (Low + High Emerging)
    # Band 2: Developing/Transitioning (Developing + Transitioning)
    # Band 3: At Grade Level (already *_at_grade_level)
    for prefix in GRADE_LANG_GROUPS:
        fe[f"{prefix}_emerging"] = fe[f"{prefix}_low_emerging"] + fe[f"{prefix}_high_emerging"]
        fe[f"{prefix}_dev_trans"] = fe[f"{prefix}_developing"] + fe[f"{prefix}_transitioning"]

    fe_cols = [
        "school_id", "region", "division", "language_clean", "lang_pair_type",
        "mt_not_assessed", "mt_fil_identical", "small_group", "n_test_takers",
        "g1_mt_at_grade_level", "g1_mt_total",
        "g2_mt_at_grade_level", "g2_mt_total",
        "g2_fil_at_grade_level", "g2_fil_total",
        "g3_mt_at_grade_level", "g3_mt_total",
        "g3_fil_at_grade_level", "g3_fil_total",
        "g3_eng_at_grade_level", "g3_eng_total",
        "g1_mt_pct", "g2_mt_pct", "g2_fil_pct", "g3_mt_pct", "g3_fil_pct", "g3_eng_pct", "early_mt_pct",
        "gap_g3_fil_early_mt", "gap_g3_eng_early_mt", "gap_g2_fil_mt", "gap_g3_fil_mt", "gap_g3_eng_mt",
    ]
    for prefix in GRADE_LANG_GROUPS:
        fe_cols.extend([f"{prefix}_emerging", f"{prefix}_dev_trans"])

    fe[fe_cols].to_csv("outputs/crla_fe.csv", index=False)

    print(f"Saved outputs/crla_cleaned.csv ({crla[cleaned_cols].shape}) and outputs/crla_fe.csv ({fe[fe_cols].shape})")
    print("Flag totals:", fe[["mt_not_assessed", "mt_fil_identical", "small_group"]].sum().to_dict())
    print("Language counts:", fe["language_clean"].value_counts().to_dict())


if __name__ == "__main__":
    main()