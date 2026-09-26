"""
Stage 0 Data Exploration — Business Entity Resolution Challenge

Answers the 5 required questions using train_source1/2/3.tsv and
train_ground_truth.tsv. Point DATA_DIR at the folder containing all four
files (e.g. dataset/train/).

Usage:
    python stage_0.py
"""
import pandas as pd
from collections import Counter
from contextlib import redirect_stdout


def load(data_dir):
    s1 = pd.read_csv(f"{data_dir}/train_source1.tsv", sep="\t", dtype=str)
    s2 = pd.read_csv(f"{data_dir}/train_source2.tsv", sep="\t", dtype=str)
    s3 = pd.read_csv(f"{data_dir}/train_source3.tsv", sep="\t", dtype=str)
    gt = pd.read_csv(f"{data_dir}/train_ground_truth.tsv", sep="\t", dtype=str)
    gt["matched_entity_ids"] = gt["matched_entity_ids"].fillna("")
    return s1, s2, s3, gt


def load_test(data_dir):
    s1 = pd.read_csv(f"{data_dir}/test_source1.tsv", sep="\t", dtype=str)
    s2 = pd.read_csv(f"{data_dir}/test_source2.tsv", sep="\t", dtype=str)
    s3 = pd.read_csv(f"{data_dir}/test_source3.tsv", sep="\t", dtype=str)
    return s1, s2, s3


def explode_gt(gt):
    """One row per (source1_entity_id, matched_id)."""
    rows = []
    for s1_id, ids in zip(gt["source1_entity_id"], gt["matched_entity_ids"]):
        if ids.strip() == "":
            continue
        for mid in ids.split(","):
            mid = mid.strip()
            if mid:
                rows.append((s1_id, mid))
    return pd.DataFrame(rows, columns=["source1_entity_id", "matched_id"])


def q1_cardinality(gt):
    print("\n=== Q1: Cardinality distribution ===")
    n_matches = gt["matched_entity_ids"].apply(
        lambda x: 0 if x.strip() == "" else len(x.split(","))
    )
    print(n_matches.describe())
    print("\nDistribution of match-count (0,1,2,3,...):")
    print(n_matches.value_counts().sort_index().head(20))
    singleton_frac = (n_matches == 0).mean()
    print(
        f"\nFraction of S1 entities with ZERO matches (singletons): {singleton_frac:.4f}")
    print(f"Mean matches per S1 entity: {n_matches.mean():.3f}")
    print(f"Median matches per S1 entity: {n_matches.median():.1f}")
    return n_matches


def q2_exclusivity(gt_long):
    print("\n=== Q2: Exclusivity (does an S2/S3 record match >1 S1 entity?) ===")
    counts = gt_long.groupby("matched_id")["source1_entity_id"].nunique()
    multi = counts[counts > 1]
    print(f"Total distinct matched S2/S3 ids: {len(counts)}")
    print(f"Matched ids that map to >1 S1 entity: {len(multi)} "
          f"({len(multi) / max(len(counts), 1):.4%})")
    if len(multi) > 0:
        print("Example violations:")
        print(multi.head(10))
    else:
        print(">>> NEVER violated in training data: safe to enforce one-to-one "
              "assignment (each S2/S3 record used by at most one S1 entity) "
              "as a hard constraint at inference time.")
    return multi


def q3_cross_country(s1, s2, s3, gt_long):
    print("\n=== Q3: Cross-country matches ===")
    country_map = pd.concat([
        s1.set_index("entity_id")["country"],
        s2.set_index("entity_id")["country"],
        s3.set_index("entity_id")["country"],
    ])
    merged = gt_long.copy()
    merged["s1_country"] = merged["source1_entity_id"].map(country_map)
    merged["match_country"] = merged["matched_id"].map(country_map)
    mismatch = merged[merged["s1_country"] != merged["match_country"]]
    print(f"Total ground-truth match pairs: {len(merged)}")
    print(
        f"Cross-country mismatches: {len(mismatch)} ({len(mismatch)/max(len(merged), 1):.4%})")
    if len(mismatch) == 0:
        print(">>> NEVER happens in training: country is safe to use as a HARD "
              "blocking key (only compare within the same country) — BUT remember "
              "test has France too, and this rule was learned only from US/India, "
              "so validate the assumption doesn't silently break recall for France.")
    else:
        print("Example cross-country matches (investigate before hard-blocking):")
        print(mismatch.head(10))
    return mismatch


def q4_dataset_size(s1, s2, s3):
    print("\n=== Q4: Dataset size per source per country ===")
    for name, df in [("Source1", s1), ("Source2", s2), ("Source3", s3)]:
        print(f"\n{name}: {len(df)} total")
        print(df["country"].value_counts())
    total_pairs_naive = 0
    print("\nNaive full cross-product size (for feasibility of LLM pass):")
    for name, df in [("Source2", s2), ("Source3", s3)]:
        naive = len(s1) * len(df)
        print(f"  S1 x {name}: {naive:,} pairs (without blocking)")


def q5_common_chains(s1, s2, s3, top_n=25):
    print(
        f"\n=== Q5: Common/high-frequency business names (top {top_n} per source) ===")
    for name, df in [("Source1", s1), ("Source2", s2), ("Source3", s3)]:
        print(f"\n{name} top names:")
        print(df["business_name"].value_counts().head(top_n))

    # Cross-source: names appearing very frequently overall -> chain candidates
    all_names = pd.concat(
        [s1["business_name"], s2["business_name"], s3["business_name"]])
    overall = all_names.value_counts()
    chains = overall[overall >= overall.quantile(0.999)]
    print(f"\nTop overall high-frequency names (candidates needing address-level "
          f"disambiguation, not name similarity):")
    print(chains.head(30))
    return chains


def q6_missing_country(s1, s2, s3, test_s1, test_s2, test_s3):
    for name, df in [
        ("train_s1", s1),
        ("train_s2", s2),
        ("train_s3", s3),
        ("test_s1", test_s1),
        ("test_s2", test_s2),
        ("test_s3", test_s3),
    ]:
        missing = df["country"].isna() | (
            df["country"].fillna("").str.strip() == "")
        print(
            f"{name}: {missing.sum()} missing country out of {len(df)} "
            f"({missing.mean():.4%})"
        )

    print("\nDistinct country values:")
    for name, df in [
        ("train_s1", s1),
        ("train_s2", s2),
        ("train_s3", s3),
        ("test_s1", test_s1),
        ("test_s2", test_s2),
        ("test_s3", test_s3),
    ]:
        print(
            f"{name} country values: {sorted(df['country'].dropna().unique())}")


def q7_test_country_counts(test_s1, test_s2, test_s3):
    for name, df in [
        ("test_s1", test_s1),
        ("test_s2", test_s2),
        ("test_s3", test_s3),
    ]:
        print(f"{name}:")
        print(df["country"].value_counts().to_string())
        print()


def run_test_country_counts():
    test_data_dir = "6ab10eb3b23ba_student_resource/student_resource/dataset/test"
    test_s1, test_s2, test_s3 = load_test(test_data_dir)
    with open("stage_0_country.txt", "w", encoding="utf-8") as country_file:
        with redirect_stdout(country_file):
            q7_test_country_counts(test_s1, test_s2, test_s3)


def main():
    data_dir = "6ab10eb3b23ba_student_resource/student_resource/dataset/train"
    test_data_dir = "6ab10eb3b23ba_student_resource/student_resource/dataset/test"
    s1, s2, s3, gt = load(data_dir)
    test_s1, test_s2, test_s3 = load_test(test_data_dir)
    gt_long = explode_gt(gt)

    q1_cardinality(gt)
    q2_exclusivity(gt_long)
    q3_cross_country(s1, s2, s3, gt_long)
    q4_dataset_size(s1, s2, s3)
    q5_common_chains(s1, s2, s3)

    with open("stage_0_country.txt", "w", encoding="utf-8") as country_file:
        with redirect_stdout(country_file):
            q6_missing_country(s1, s2, s3, test_s1, test_s2, test_s3)


if __name__ == "__main__":
    run_test_country_counts()
