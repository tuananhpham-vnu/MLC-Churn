"""Improved EDA and preprocessing report for the MLC churn dataset.

This script extends the baseline EDA with:
- publication-ready visualizations
- grouped churn comparisons
- statistical tests for numeric and categorical features
- outlier diagnostics
- a feature-selected model-ready dataset

Run from the project root or from this folder:
    python Phase_0_EDA/00_eda_improved.py
"""

from __future__ import annotations

import math
from pathlib import Path
from textwrap import dedent

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


RANDOM_SEED = 1234
TARGET = "churn"
POSITIVE_CLASS = "yes"
HIGH_CORRELATION_THRESHOLD = 0.95
MIN_CATEGORY_COUNT_FOR_RANKING = 20


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


ROOT_DIR = project_root()
DATA_PATH = ROOT_DIR / "mlc_churn.csv"
PHASE_DIR = ROOT_DIR / "Phase_0_EDA"
OUTPUT_DIR = PHASE_DIR / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"


def section(title: str) -> None:
    line = "=" * 80
    print(f"\n{line}\n{title}\n{line}")


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)


def normalize_categories(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    normalized = df.copy()
    for column in columns:
        normalized[column] = normalized[column].astype(str).str.strip().str.lower()
    return normalized


def split_feature_types(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    features = df.drop(columns=[TARGET], errors="ignore")
    numeric_features = features.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = features.select_dtypes(exclude=[np.number]).columns.tolist()
    return numeric_features, categorical_features


def encode_target(df: pd.DataFrame) -> pd.Series:
    return df[TARGET].astype(str).str.lower().eq(POSITIVE_CLASS).astype(int)


def save_current_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


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
    out = pd.DataFrame({"count": counts, "percent": counts / len(df) * 100})
    out.index.name = TARGET
    return out


def make_correlation_pairs(corr: pd.DataFrame) -> pd.DataFrame:
    rows = []
    columns = corr.columns.tolist()
    for i, left in enumerate(columns):
        for right in columns[i + 1 :]:
            value = corr.loc[left, right]
            rows.append(
                {
                    "feature_1": left,
                    "feature_2": right,
                    "correlation": value,
                    "abs_correlation": abs(value),
                }
            )
    return pd.DataFrame(rows).sort_values("abs_correlation", ascending=False)


def infer_redundant_charge_features(high_corr_pairs: pd.DataFrame) -> list[str]:
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


def make_outlier_report(df: pd.DataFrame, numeric_features: list[str]) -> pd.DataFrame:
    rows = []
    for column in numeric_features:
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask = (df[column] < lower) | (df[column] > upper)
        rows.append(
            {
                "feature": column,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "lower_bound": lower,
                "upper_bound": upper,
                "outlier_count": int(mask.sum()),
                "outlier_percent": mask.mean() * 100,
            }
        )
    return pd.DataFrame(rows).sort_values("outlier_count", ascending=False)


def make_data_quality_report(
    df: pd.DataFrame,
    missing_report: pd.DataFrame,
    outlier_report: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    duplicate_rows = int(df.duplicated().sum())
    for column in df.columns:
        outlier_count = 0
        if column in outlier_report["feature"].values:
            outlier_count = int(outlier_report.loc[outlier_report["feature"] == column, "outlier_count"].iloc[0])
        rows.append(
            {
                "feature": column,
                "dtype": str(df[column].dtype),
                "n_unique": int(df[column].nunique(dropna=True)),
                "missing_count": int(missing_report.loc[column, "missing_count"]),
                "missing_percent": float(missing_report.loc[column, "missing_percent"]),
                "outlier_count_iqr": outlier_count,
                "is_constant": df[column].nunique(dropna=False) <= 1,
                "duplicate_rows_in_dataset": duplicate_rows,
            }
        )
    return pd.DataFrame(rows)


def numeric_target_correlation(df: pd.DataFrame, numeric_features: list[str]) -> pd.DataFrame:
    target_binary = encode_target(df)
    rows = []
    for feature in numeric_features:
        corr = df[feature].corr(target_binary)
        rows.append(
            {
                "feature": feature,
                "correlation_with_churn_yes": corr,
                "abs_correlation": abs(corr),
            }
        )
    return pd.DataFrame(rows).sort_values("abs_correlation", ascending=False)


def numeric_group_tests(df: pd.DataFrame, numeric_features: list[str]) -> pd.DataFrame:
    rows = []
    positive = df[TARGET].astype(str).str.lower().eq(POSITIVE_CLASS)

    for feature in numeric_features:
        no_values = df.loc[~positive, feature].dropna()
        yes_values = df.loc[positive, feature].dropna()

        if len(no_values) < 2 or len(yes_values) < 2:
            continue

        no_mean = no_values.mean()
        yes_mean = yes_values.mean()
        pooled_std = math.sqrt((no_values.var(ddof=1) + yes_values.var(ddof=1)) / 2)
        cohens_d = (yes_mean - no_mean) / pooled_std if pooled_std > 0 else np.nan

        t_stat, t_pvalue = stats.ttest_ind(yes_values, no_values, equal_var=False)
        try:
            u_stat, u_pvalue = stats.mannwhitneyu(yes_values, no_values, alternative="two-sided")
        except ValueError:
            u_stat, u_pvalue = np.nan, np.nan

        rows.append(
            {
                "feature": feature,
                "mean_no": no_mean,
                "mean_yes": yes_mean,
                "mean_difference_yes_minus_no": yes_mean - no_mean,
                "median_no": no_values.median(),
                "median_yes": yes_values.median(),
                "cohens_d": cohens_d,
                "welch_t_stat": t_stat,
                "welch_pvalue": t_pvalue,
                "mannwhitney_u_stat": u_stat,
                "mannwhitney_pvalue": u_pvalue,
            }
        )

    return pd.DataFrame(rows).sort_values("mannwhitney_pvalue")


def categorical_churn_rates(df: pd.DataFrame, categorical_features: list[str]) -> pd.DataFrame:
    rows = []
    for feature in categorical_features:
        grouped = (
            df.groupby(feature, dropna=False)[TARGET]
            .agg(
                total="size",
                churn_yes=lambda values: (values.astype(str).str.lower() == POSITIVE_CLASS).sum(),
            )
            .reset_index()
            .rename(columns={feature: "category"})
        )
        grouped["churn_no"] = grouped["total"] - grouped["churn_yes"]
        grouped["churn_rate"] = grouped["churn_yes"] / grouped["total"]
        grouped["feature"] = feature
        rows.append(grouped[["feature", "category", "total", "churn_no", "churn_yes", "churn_rate"]])
    if not rows:
        return pd.DataFrame(columns=["feature", "category", "total", "churn_no", "churn_yes", "churn_rate"])
    return pd.concat(rows, ignore_index=True).sort_values(
        ["feature", "churn_rate", "total"], ascending=[True, False, False]
    )


def categorical_chi_square_tests(df: pd.DataFrame, categorical_features: list[str]) -> pd.DataFrame:
    rows = []
    for feature in categorical_features:
        contingency = pd.crosstab(df[feature], df[TARGET])
        if contingency.shape[0] < 2 or contingency.shape[1] < 2:
            continue
        chi2, pvalue, dof, expected = stats.chi2_contingency(contingency)
        n = contingency.to_numpy().sum()
        min_dim = min(contingency.shape) - 1
        cramers_v = math.sqrt(chi2 / (n * min_dim)) if n > 0 and min_dim > 0 else np.nan
        rows.append(
            {
                "feature": feature,
                "chi2": chi2,
                "pvalue": pvalue,
                "dof": dof,
                "cramers_v": cramers_v,
                "min_expected_count": float(np.min(expected)),
            }
        )
    return pd.DataFrame(rows).sort_values("pvalue")


def plot_missing_values(missing_report: pd.DataFrame, path: Path) -> None:
    plot_df = missing_report.sort_values("missing_percent", ascending=True)
    plt.figure(figsize=(10, max(5, len(plot_df) * 0.28)))
    plt.barh(plot_df.index, plot_df["missing_percent"], color="#4C78A8")
    plt.xlabel("Missing values (%)")
    plt.ylabel("Feature")
    plt.title("Missing-value profile")
    if plot_df["missing_percent"].max() == 0:
        plt.text(
            0.5,
            0.5,
            "No missing values detected",
            transform=plt.gca().transAxes,
            ha="center",
            va="center",
            fontsize=12,
            color="#333333",
        )
        plt.xlim(0, 1)
    save_current_figure(path)


def plot_churn_distribution(distribution: pd.DataFrame, path: Path) -> None:
    plot_df = distribution.reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), gridspec_kw={"width_ratios": [1.25, 1]})

    sns.barplot(data=plot_df, x=TARGET, y="count", ax=axes[0], palette=["#4C78A8", "#F58518"], hue=TARGET, legend=False)
    axes[0].set_title("Churn class counts")
    axes[0].set_xlabel("Churn")
    axes[0].set_ylabel("Count")
    for container in axes[0].containers:
        axes[0].bar_label(container, fmt="%d")

    axes[1].pie(
        plot_df["count"],
        labels=plot_df[TARGET],
        autopct="%1.1f%%",
        colors=["#4C78A8", "#F58518"],
        startangle=90,
        textprops={"fontsize": 10},
    )
    axes[1].set_title("Churn class share")
    save_current_figure(path)


def plot_correlation_heatmap(correlation_matrix: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(
        correlation_matrix,
        mask=mask,
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.35,
        cbar_kws={"label": "Pearson correlation"},
    )
    plt.title("Numeric-feature correlation heatmap")
    save_current_figure(path)


def plot_numeric_distributions_by_churn(
    df: pd.DataFrame, numeric_features: list[str], path: Path
) -> None:
    cols = 3
    rows = math.ceil(len(numeric_features) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(15, max(4, rows * 3.2)))
    axes_flat = np.array(axes).reshape(-1)

    for ax, feature in zip(axes_flat, numeric_features):
        sns.boxplot(data=df, x=TARGET, y=feature, ax=ax, palette=["#4C78A8", "#F58518"], hue=TARGET, legend=False)
        ax.set_title(feature)
        ax.set_xlabel("Churn")
        ax.set_ylabel("")

    for ax in axes_flat[len(numeric_features) :]:
        ax.axis("off")

    fig.suptitle("Numeric feature distributions by churn", y=1.01, fontsize=14)
    save_current_figure(path)


def plot_top_target_correlations(target_corr: pd.DataFrame, path: Path) -> None:
    plot_df = target_corr.head(12).sort_values("correlation_with_churn_yes")
    colors = np.where(plot_df["correlation_with_churn_yes"] >= 0, "#F58518", "#4C78A8")

    plt.figure(figsize=(10, 6))
    plt.barh(plot_df["feature"], plot_df["correlation_with_churn_yes"], color=colors)
    plt.axvline(0, color="#333333", linewidth=0.8)
    plt.xlabel("Pearson correlation with churn=yes")
    plt.ylabel("Feature")
    plt.title("Top numeric associations with churn")
    save_current_figure(path)


def plot_categorical_churn_rates(rates: pd.DataFrame, path: Path) -> None:
    key_features = ["international_plan", "voice_mail_plan", "area_code"]
    plot_df = rates[
        rates["feature"].isin(key_features)
        & (rates["total"] >= MIN_CATEGORY_COUNT_FOR_RANKING)
    ].copy()

    if plot_df.empty:
        return

    plot_df["label"] = plot_df["feature"] + "=" + plot_df["category"].astype(str)
    plot_df = plot_df.sort_values("churn_rate", ascending=True)

    plt.figure(figsize=(9, 5))
    plt.barh(plot_df["label"], plot_df["churn_rate"] * 100, color="#54A24B")
    plt.xlabel("Churn rate (%)")
    plt.ylabel("Category")
    plt.title("Churn rate by key categorical features")
    save_current_figure(path)


def plot_state_churn_rates(rates: pd.DataFrame, path: Path) -> None:
    state_rates = rates[
        (rates["feature"] == "state") & (rates["total"] >= MIN_CATEGORY_COUNT_FOR_RANKING)
    ].copy()
    if state_rates.empty:
        return

    top_bottom = pd.concat(
        [
            state_rates.nsmallest(10, "churn_rate"),
            state_rates.nlargest(10, "churn_rate"),
        ],
        ignore_index=True,
    ).drop_duplicates(subset=["category"])
    top_bottom = top_bottom.sort_values("churn_rate", ascending=True)

    plt.figure(figsize=(9, 7))
    plt.barh(top_bottom["category"], top_bottom["churn_rate"] * 100, color="#B279A2")
    plt.xlabel("Churn rate (%)")
    plt.ylabel("State")
    plt.title("Lowest and highest state-level churn rates")
    save_current_figure(path)


def plot_outlier_summary(outlier_report: pd.DataFrame, path: Path) -> None:
    plot_df = outlier_report.sort_values("outlier_percent", ascending=True)
    plt.figure(figsize=(10, 6))
    plt.barh(plot_df["feature"], plot_df["outlier_percent"], color="#E45756")
    plt.xlabel("IQR outlier rate (%)")
    plt.ylabel("Feature")
    plt.title("Outlier screen by numeric feature")
    save_current_figure(path)


def build_selected_model_ready_data(
    df: pd.DataFrame,
    categorical_features: list[str],
    redundant_features: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    working = df.copy()
    working[f"{TARGET}_binary"] = encode_target(working)

    drop_features = sorted(set(redundant_features))
    feature_df = working.drop(columns=[TARGET] + drop_features)
    feature_df = pd.get_dummies(feature_df, columns=categorical_features, drop_first=False, dtype=int)

    ordered_columns = [column for column in feature_df.columns if column != f"{TARGET}_binary"]
    ordered_columns = ordered_columns + [f"{TARGET}_binary"]
    return feature_df[ordered_columns], drop_features


def write_preprocessing_report(
    path: Path,
    df: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    churn_distribution: pd.DataFrame,
    missing_report: pd.DataFrame,
    high_corr_pairs: pd.DataFrame,
    redundant_features: list[str],
    outlier_report: pd.DataFrame,
    numeric_tests: pd.DataFrame,
    categorical_tests: pd.DataFrame,
    selected_shape: tuple[int, int],
) -> None:
    churn_yes_count = int(churn_distribution.loc[POSITIVE_CLASS, "count"])
    churn_yes_pct = float(churn_distribution.loc[POSITIVE_CLASS, "percent"])
    total_missing = int(missing_report["missing_count"].sum())
    duplicate_rows = int(df.duplicated().sum())

    top_numeric = numeric_tests.head(8)[
        [
            "feature",
            "mean_no",
            "mean_yes",
            "mean_difference_yes_minus_no",
            "cohens_d",
            "mannwhitney_pvalue",
        ]
    ].round(6)
    top_categorical = categorical_tests.head(8).round(6)

    lines = [
        "MLC Churn - Phase 0 Improved EDA Report",
        "=" * 52,
        "",
        "Dataset summary",
        "-" * 20,
        f"Rows: {df.shape[0]}",
        f"Columns: {df.shape[1]}",
        f"Numeric predictors: {len(numeric_features)}",
        f"Categorical predictors: {len(categorical_features)}",
        f"Target: {TARGET}; positive class: {POSITIVE_CLASS}",
        f"churn=yes count: {churn_yes_count} ({churn_yes_pct:.2f}%)",
        f"Total missing cells: {total_missing}",
        f"Duplicate rows: {duplicate_rows}",
        "",
        "Important EDA findings",
        "-" * 20,
        "1. The target is imbalanced, so every train/test split and CV fold should be stratified.",
        "2. There are no missing values in the current CSV, but the preprocessing plan still defines future imputers.",
        "3. Charge columns are almost perfectly correlated with their matching minutes columns.",
        "4. Keep minutes and consider dropping charges to reduce redundant signal.",
        "5. customer_service_calls, international_plan, and usage minutes are expected to be useful churn signals.",
        "",
        "High-correlation pairs",
        "-" * 20,
        high_corr_pairs.round(6).to_string(index=False) if not high_corr_pairs.empty else "None",
        "",
        "Recommended drop list",
        "-" * 20,
        ", ".join(redundant_features) if redundant_features else "None",
        "",
        "Outlier screen (top 10 by count)",
        "-" * 20,
        outlier_report.head(10).round(4).to_string(index=False),
        "",
        "Top numeric churn comparisons",
        "-" * 20,
        top_numeric.to_string(index=False),
        "",
        "Categorical chi-square tests",
        "-" * 20,
        top_categorical.to_string(index=False),
        "",
        "Preprocessing recipe for Phase 1+",
        "-" * 20,
        dedent(
            f"""
            1. Load mlc_churn.csv and keep it as the raw source of truth.
            2. Strip/lowercase categorical values: {', '.join(categorical_features)}.
            3. Encode target as churn_binary = 1 for yes and 0 for no.
            4. Missing values if new data has them:
               - numeric: median imputation
               - categorical: most-frequent imputation
            5. One-hot encode categorical predictors with unknown-category handling in modeling code.
            6. Drop redundant charge columns selected from correlation >= {HIGH_CORRELATION_THRESHOLD}: {', '.join(redundant_features) if redundant_features else 'none'}.
            7. Use StratifiedKFold/stratified train_test_split because churn=yes is {churn_yes_pct:.2f}%.
            8. Set random_state={RANDOM_SEED} in all random steps.
            """
        ).strip(),
        "",
        f"Selected model-ready dataset shape: {selected_shape[0]} rows x {selected_shape[1]} columns",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")

    raw_df = load_data(DATA_PATH)
    numeric_features, categorical_features = split_feature_types(raw_df)
    df = normalize_categories(raw_df, categorical_features + [TARGET])

    section("1. Improved EDA setup")
    print(f"Loaded: {DATA_PATH}")
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Numeric predictors: {len(numeric_features)}")
    print(f"Categorical predictors: {len(categorical_features)}")

    section("2. Data quality")
    missing_report = make_missing_report(df)
    outlier_report = make_outlier_report(df, numeric_features)
    quality_report = make_data_quality_report(df, missing_report, outlier_report)

    missing_report.to_csv(OUTPUT_DIR / "improved_missing_values.csv")
    outlier_report.to_csv(OUTPUT_DIR / "outlier_report.csv", index=False)
    quality_report.to_csv(OUTPUT_DIR / "data_quality_report.csv", index=False)

    print(f"Total missing cells: {int(missing_report['missing_count'].sum())}")
    print(f"Duplicate rows: {int(df.duplicated().sum())}")
    print("\nTop outlier counts:")
    print(outlier_report.head(10).round(4).to_string(index=False))

    section("3. Class balance and correlations")
    churn_distribution = make_churn_distribution(df)
    correlation_matrix = df[numeric_features].corr()
    correlation_pairs = make_correlation_pairs(correlation_matrix)
    high_corr_pairs = correlation_pairs[
        correlation_pairs["abs_correlation"] >= HIGH_CORRELATION_THRESHOLD
    ].copy()
    redundant_features = infer_redundant_charge_features(high_corr_pairs)
    target_corr = numeric_target_correlation(df, numeric_features)

    churn_distribution.to_csv(OUTPUT_DIR / "improved_churn_distribution.csv")
    correlation_matrix.to_csv(OUTPUT_DIR / "improved_correlation_matrix.csv")
    correlation_pairs.to_csv(OUTPUT_DIR / "improved_correlation_pairs.csv", index=False)
    high_corr_pairs.to_csv(OUTPUT_DIR / "improved_high_correlation_pairs.csv", index=False)
    target_corr.to_csv(OUTPUT_DIR / "improved_numeric_target_correlation.csv", index=False)

    print(churn_distribution.round(4).to_string())
    print("\nHigh-correlation pairs:")
    print(high_corr_pairs.round(6).to_string(index=False))
    print("\nRecommended redundant charge features:")
    print(", ".join(redundant_features) if redundant_features else "None")

    section("4. Statistical comparisons")
    rates = categorical_churn_rates(df, categorical_features)
    numeric_tests = numeric_group_tests(df, numeric_features)
    categorical_tests = categorical_chi_square_tests(df, categorical_features)
    group_summary = df.groupby(TARGET)[numeric_features].agg(["mean", "median", "std"]).round(4)

    rates.to_csv(OUTPUT_DIR / "improved_categorical_churn_rates.csv", index=False)
    numeric_tests.to_csv(OUTPUT_DIR / "numeric_group_tests.csv", index=False)
    categorical_tests.to_csv(OUTPUT_DIR / "categorical_chi_square_tests.csv", index=False)
    group_summary.to_csv(OUTPUT_DIR / "improved_numeric_summary_by_churn.csv")

    print("\nTop numeric tests by Mann-Whitney p-value:")
    print(
        numeric_tests.head(10)[
            [
                "feature",
                "mean_no",
                "mean_yes",
                "mean_difference_yes_minus_no",
                "cohens_d",
                "mannwhitney_pvalue",
            ]
        ]
        .round(6)
        .to_string(index=False)
    )
    print("\nCategorical chi-square tests:")
    print(categorical_tests.round(6).to_string(index=False))

    section("5. Figures")
    plot_missing_values(missing_report, FIGURE_DIR / "missing_values.png")
    plot_churn_distribution(churn_distribution, FIGURE_DIR / "churn_distribution.png")
    plot_correlation_heatmap(correlation_matrix, FIGURE_DIR / "correlation_heatmap.png")
    plot_numeric_distributions_by_churn(df, numeric_features, FIGURE_DIR / "numeric_distributions_by_churn.png")
    plot_top_target_correlations(target_corr, FIGURE_DIR / "top_numeric_target_correlations.png")
    plot_categorical_churn_rates(rates, FIGURE_DIR / "categorical_churn_rates.png")
    plot_state_churn_rates(rates, FIGURE_DIR / "state_churn_rates.png")
    plot_outlier_summary(outlier_report, FIGURE_DIR / "outlier_summary.png")
    print(f"Saved figures to: {FIGURE_DIR}")

    section("6. Improved preprocessing output")
    selected_df, dropped_features = build_selected_model_ready_data(
        df, categorical_features, redundant_features
    )
    selected_path = OUTPUT_DIR / "preprocessed_improved.csv"
    selected_df.to_csv(selected_path, index=False)

    feature_list_path = OUTPUT_DIR / "selected_feature_list.txt"
    selected_feature_names = [column for column in selected_df.columns if column != f"{TARGET}_binary"]
    feature_list_path.write_text("\n".join(selected_feature_names), encoding="utf-8")

    write_preprocessing_report(
        OUTPUT_DIR / "improved_eda_report.txt",
        df,
        numeric_features,
        categorical_features,
        churn_distribution,
        missing_report,
        high_corr_pairs,
        redundant_features,
        outlier_report,
        numeric_tests,
        categorical_tests,
        selected_df.shape,
    )

    print(f"Saved selected model-ready data: {selected_path}")
    print(f"Shape: {selected_df.shape[0]} rows x {selected_df.shape[1]} columns")
    print(f"Dropped redundant features: {', '.join(dropped_features) if dropped_features else 'None'}")

    section("Done")
    print(f"All improved EDA outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
