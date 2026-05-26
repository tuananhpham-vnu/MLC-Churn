"""Baseline EDA for the MLC churn dataset.

This script focuses on reproducible tabular outputs:
- dataset structure and basic statistics
- missing-value analysis
- churn class balance
- numeric correlations and redundant-feature candidates
- a simple, model-ready encoded dataset for the next phases

Run from the project root or from this folder:
    python Phase_0_EDA/00_eda_baseline.py
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd


RANDOM_SEED = 1234
TARGET = "churn"
POSITIVE_CLASS = "yes"
HIGH_CORRELATION_THRESHOLD = 0.95


def project_root() -> Path:
    """Return the repository root regardless of the current working directory."""
    return Path(__file__).resolve().parents[1]


ROOT_DIR = project_root()
DATA_PATH = ROOT_DIR / "mlc_churn.csv"
PHASE_DIR = ROOT_DIR / "Phase_0_EDA"
OUTPUT_DIR = PHASE_DIR / "outputs"


def section(title: str) -> None:
    line = "=" * 80
    print(f"\n{line}\n{title}\n{line}")


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)


def split_feature_types(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    feature_df = df.drop(columns=[TARGET], errors="ignore")
    numeric_features = feature_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = feature_df.select_dtypes(exclude=[np.number]).columns.tolist()
    return numeric_features, categorical_features


def make_missing_report(df: pd.DataFrame) -> pd.DataFrame:
    report = pd.DataFrame(
        {
            "missing_count": df.isna().sum(),
            "missing_percent": df.isna().mean() * 100,
            "dtype": df.dtypes.astype(str),
            "n_unique": df.nunique(dropna=True),
        }
    )
    return report.sort_values(["missing_count", "missing_percent"], ascending=False)


def make_churn_distribution(df: pd.DataFrame) -> pd.DataFrame:
    counts = df[TARGET].value_counts(dropna=False)
    distribution = pd.DataFrame(
        {
            "count": counts,
            "percent": counts / len(df) * 100,
        }
    )
    distribution.index.name = TARGET
    return distribution


def make_correlation_pairs(correlation_matrix: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    columns = correlation_matrix.columns.tolist()
    for idx, left in enumerate(columns):
        for right in columns[idx + 1 :]:
            value = correlation_matrix.loc[left, right]
            rows.append(
                {
                    "feature_1": left,
                    "feature_2": right,
                    "correlation": value,
                    "abs_correlation": abs(value),
                }
            )
    return pd.DataFrame(rows).sort_values("abs_correlation", ascending=False)


def encode_target(df: pd.DataFrame) -> pd.Series:
    return df[TARGET].astype(str).str.lower().eq(POSITIVE_CLASS).astype(int)


def make_numeric_target_correlation(
    df: pd.DataFrame, numeric_features: list[str]
) -> pd.DataFrame:
    target_binary = encode_target(df)
    rows = []
    for column in numeric_features:
        corr = df[column].corr(target_binary)
        rows.append(
            {
                "feature": column,
                "correlation_with_churn_yes": corr,
                "abs_correlation": abs(corr),
            }
        )
    return pd.DataFrame(rows).sort_values("abs_correlation", ascending=False)


def make_categorical_churn_rates(
    df: pd.DataFrame, categorical_features: list[str]
) -> pd.DataFrame:
    rows = []
    for column in categorical_features:
        grouped = (
            df.groupby(column, dropna=False)[TARGET]
            .agg(
                total="size",
                churn_yes=lambda values: (values.astype(str).str.lower() == POSITIVE_CLASS).sum(),
            )
            .reset_index()
        )
        grouped["churn_rate"] = grouped["churn_yes"] / grouped["total"]
        grouped["feature"] = column
        grouped = grouped.rename(columns={column: "category"})
        rows.append(grouped[["feature", "category", "total", "churn_yes", "churn_rate"]])

    if not rows:
        return pd.DataFrame(columns=["feature", "category", "total", "churn_yes", "churn_rate"])

    return pd.concat(rows, ignore_index=True).sort_values(
        ["feature", "churn_rate", "total"], ascending=[True, False, False]
    )


def make_group_summary(df: pd.DataFrame, numeric_features: list[str]) -> pd.DataFrame:
    if not numeric_features:
        return pd.DataFrame()
    return df.groupby(TARGET)[numeric_features].agg(["mean", "median", "std"]).round(4)


def infer_redundant_charge_features(high_corr_pairs: pd.DataFrame) -> list[str]:
    """Prefer minutes over charge when both are almost perfectly correlated."""
    redundant = set()
    for _, row in high_corr_pairs.iterrows():
        left = str(row["feature_1"])
        right = str(row["feature_2"])
        if row["abs_correlation"] < HIGH_CORRELATION_THRESHOLD:
            continue
        if left.endswith("_charge") and right.endswith("_minutes"):
            redundant.add(left)
        elif right.endswith("_charge") and left.endswith("_minutes"):
            redundant.add(right)
    return sorted(redundant)


def make_model_ready_dataset(
    df: pd.DataFrame,
    categorical_features: list[str],
    redundant_features: list[str],
) -> pd.DataFrame:
    """Create a simple encoded dataset while preserving reproducibility.

    The improved script creates a feature-selected version. Baseline keeps all
    non-target predictors so later phases can decide which variables to remove.
    """
    clean_df = df.copy()

    for column in categorical_features + [TARGET]:
        clean_df[column] = clean_df[column].astype(str).str.strip().str.lower()

    clean_df[f"{TARGET}_binary"] = clean_df[TARGET].eq(POSITIVE_CLASS).astype(int)
    model_df = clean_df.drop(columns=[TARGET])
    model_df = pd.get_dummies(model_df, columns=categorical_features, drop_first=False, dtype=int)
    ordered_columns = [column for column in model_df.columns if column != f"{TARGET}_binary"]
    model_df = model_df[ordered_columns + [f"{TARGET}_binary"]]

    # Keep the redundant columns in the baseline file, but store an explicit flag
    # in a companion text report so Phase 1 can choose a feature-selection policy.
    model_df.attrs["redundant_features"] = redundant_features
    return model_df


def write_feature_summary(
    output_path: Path,
    df: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    churn_distribution: pd.DataFrame,
    missing_report: pd.DataFrame,
    high_corr_pairs: pd.DataFrame,
    redundant_features: list[str],
) -> None:
    lines = [
        "MLC Churn - Phase 0 Baseline EDA Summary",
        "=" * 48,
        "",
        f"Dataset path: {DATA_PATH}",
        f"Rows: {df.shape[0]}",
        f"Columns: {df.shape[1]}",
        f"Target column: {TARGET}",
        f"Random seed for later modeling: {RANDOM_SEED}",
        "",
        "Feature types",
        "-" * 20,
        f"Numeric features ({len(numeric_features)}): {', '.join(numeric_features)}",
        f"Categorical features ({len(categorical_features)}): {', '.join(categorical_features)}",
        "",
        "Class balance",
        "-" * 20,
        churn_distribution.round(4).to_string(),
        "",
        "Missing values",
        "-" * 20,
        f"Total missing cells: {int(missing_report['missing_count'].sum())}",
        "Columns with missing values:",
    ]

    missing_columns = missing_report[missing_report["missing_count"] > 0]
    if missing_columns.empty:
        lines.append("None. No imputation is needed for the current file.")
    else:
        lines.append(missing_columns.to_string())

    lines.extend(
        [
            "",
            "High-correlation feature pairs",
            "-" * 20,
        ]
    )

    top_pairs = high_corr_pairs.head(12)
    lines.append(top_pairs.round(6).to_string(index=False))

    lines.extend(
        [
            "",
            "Redundant-feature recommendation",
            "-" * 20,
        ]
    )
    if redundant_features:
        lines.append(
            "Candidate drop list because charge columns are deterministic or near-deterministic "
            f"from minutes: {', '.join(redundant_features)}"
        )
    else:
        lines.append("No feature pair exceeded the configured high-correlation threshold.")

    lines.extend(
        [
            "",
            "Preprocessing plan for Phase 1",
            "-" * 20,
            "1. Keep the raw CSV as the source of truth.",
            "2. Normalize categorical strings with strip/lowercase.",
            "3. Encode churn as 1 for yes and 0 for no.",
            "4. One-hot encode state, area_code, international_plan, and voice_mail_plan.",
            "5. Use stratified train/test split or stratified cross-validation because churn=yes is the minority class.",
            "6. Consider dropping charge columns after correlation analysis, keeping the corresponding minutes columns.",
            "7. Use random_state=1234 for every split and model initialization.",
        ]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data(DATA_PATH)
    numeric_features, categorical_features = split_feature_types(df)

    section("1. Dataset overview")
    print(f"Data path: {DATA_PATH}")
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print("\nColumns:")
    print(pd.DataFrame({"column": df.columns, "dtype": df.dtypes.astype(str).values}).to_string(index=False))
    print("\nHead:")
    print(df.head().to_string(index=False))
    print("\nTail:")
    print(df.tail().to_string(index=False))

    info_buffer = io.StringIO()
    df.info(buf=info_buffer)
    (OUTPUT_DIR / "data_info.txt").write_text(info_buffer.getvalue(), encoding="utf-8")

    section("2. Summary statistics")
    numeric_summary = df[numeric_features].describe().T
    categorical_summary = df[categorical_features + [TARGET]].describe().T
    print("\nNumeric summary:")
    print(numeric_summary.round(4).to_string())
    print("\nCategorical summary:")
    print(categorical_summary.to_string())
    numeric_summary.to_csv(OUTPUT_DIR / "numeric_summary.csv")
    categorical_summary.to_csv(OUTPUT_DIR / "categorical_summary.csv")

    section("3. Missing values")
    missing_report = make_missing_report(df)
    print(missing_report.to_string())
    missing_report.to_csv(OUTPUT_DIR / "missing_values.csv")

    section("4. Churn class distribution")
    churn_distribution = make_churn_distribution(df)
    print(churn_distribution.round(4).to_string())
    churn_distribution.to_csv(OUTPUT_DIR / "churn_distribution.csv")

    churn_yes_rate = churn_distribution.loc[POSITIVE_CLASS, "percent"] if POSITIVE_CLASS in churn_distribution.index else 0
    if churn_yes_rate < 20:
        print(f"\nClass imbalance detected: churn=yes is only {churn_yes_rate:.2f}% of the data.")
    else:
        print(f"\nClass balance is moderate: churn=yes is {churn_yes_rate:.2f}% of the data.")

    section("5. Feature type and cardinality")
    feature_type_summary = pd.DataFrame(
        {
            "feature": numeric_features + categorical_features,
            "type": ["numeric"] * len(numeric_features) + ["categorical"] * len(categorical_features),
            "n_unique": [df[column].nunique(dropna=True) for column in numeric_features + categorical_features],
        }
    )
    print(feature_type_summary.to_string(index=False))
    feature_type_summary.to_csv(OUTPUT_DIR / "feature_type_summary.csv", index=False)

    section("6. Correlation analysis")
    correlation_matrix = df[numeric_features].corr()
    correlation_matrix.to_csv(OUTPUT_DIR / "correlation_matrix.csv")
    correlation_pairs = make_correlation_pairs(correlation_matrix)
    correlation_pairs.to_csv(OUTPUT_DIR / "correlation_pairs.csv", index=False)

    high_corr_pairs = correlation_pairs[
        correlation_pairs["abs_correlation"] >= HIGH_CORRELATION_THRESHOLD
    ].copy()
    high_corr_pairs.to_csv(OUTPUT_DIR / "high_correlation_pairs.csv", index=False)
    print("Top absolute correlations:")
    print(correlation_pairs.head(12).round(6).to_string(index=False))
    print(f"\nPairs with abs(correlation) >= {HIGH_CORRELATION_THRESHOLD}:")
    if high_corr_pairs.empty:
        print("None")
    else:
        print(high_corr_pairs.round(6).to_string(index=False))

    section("7. Target association")
    numeric_target_corr = make_numeric_target_correlation(df, numeric_features)
    categorical_churn_rates = make_categorical_churn_rates(df, categorical_features)
    group_summary = make_group_summary(df, numeric_features)

    print("\nNumeric correlation with churn=yes:")
    print(numeric_target_corr.round(6).to_string(index=False))
    numeric_target_corr.to_csv(OUTPUT_DIR / "numeric_target_correlation.csv", index=False)
    categorical_churn_rates.to_csv(OUTPUT_DIR / "categorical_churn_rates.csv", index=False)
    group_summary.to_csv(OUTPUT_DIR / "numeric_summary_by_churn.csv")

    print("\nHighest categorical churn-rate groups with at least 20 records:")
    filtered_rates = categorical_churn_rates[categorical_churn_rates["total"] >= 20]
    print(filtered_rates.head(15).round(4).to_string(index=False))

    section("8. Baseline preprocessing output")
    redundant_features = infer_redundant_charge_features(high_corr_pairs)
    model_ready_df = make_model_ready_dataset(df, categorical_features, redundant_features)
    model_ready_path = OUTPUT_DIR / "preprocessed_baseline.csv"
    model_ready_df.to_csv(model_ready_path, index=False)

    write_feature_summary(
        OUTPUT_DIR / "feature_summary.txt",
        df,
        numeric_features,
        categorical_features,
        churn_distribution,
        missing_report,
        high_corr_pairs,
        redundant_features,
    )

    print(f"Saved model-ready baseline data: {model_ready_path}")
    print(f"Preprocessed shape: {model_ready_df.shape[0]} rows x {model_ready_df.shape[1]} columns")
    if redundant_features:
        print(f"Recommended redundant features to review/drop later: {', '.join(redundant_features)}")

    section("Done")
    print(f"All baseline EDA outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
