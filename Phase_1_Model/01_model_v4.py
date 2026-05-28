"""Phase 1 model v3: RF + feature engineering from v2 base features.

Policy:
- same RF core as baseline/v1/v2
- start from v2 base feature set (13 features)
- add engineered features (aggregate/ratio/interaction)
- run multiple max_depth values (default: 3, 5, 7, 10, None)
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split


POSITIVE_CLASS = 1
EPS = 1e-6


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


# ===== KHU VUC CONFIG =====
ROOT_DIR = project_root()
DEFAULT_INPUT = ROOT_DIR / "Phase_0_EDA" / "outputs" / "preprocessed_improved.csv"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "Phase_1_Model" / "outputs"
# DEFAULT_MAX_DEPTHS = "3,5,None"
DEFAULT_MAX_DEPTHS = "3,5,7,10,None"

BASE_FEATURES_V2 = [
    "account_length",
    "number_vmail_messages",
    "total_day_minutes",
    "total_day_calls",
    "total_eve_minutes",
    "total_eve_calls",
    "total_night_minutes",
    "total_night_calls",
    "total_intl_minutes",
    "total_intl_calls",
    "number_customer_service_calls",
    "international_plan_yes",
    "voice_mail_plan_yes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RF model v3 (feature engineering).")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--random-seed", type=int, default=1234)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--max-depths", type=str, default=DEFAULT_MAX_DEPTHS)
    return parser.parse_args()


def parse_max_depths(raw_value: str) -> list[int | None]:
    values: list[int | None] = []
    for token in raw_value.split(","):
        token = token.strip()
        if not token:
            continue
        if token.lower() in {"none", "null"}:
            value: int | None = None
        else:
            value = int(token)
            if value <= 0:
                raise ValueError("max_depth must be positive integer or None.")
        if value not in values:
            values.append(value)
    if not values:
        raise ValueError("No valid max_depth value was provided.")
    return values


def depth_tag(max_depth: int | None) -> str:
    return "none" if max_depth is None else str(max_depth)


def ensure_binary_target(df: pd.DataFrame) -> pd.Series:
    if "churn_binary" in df.columns:
        return df["churn_binary"].astype(int)
    if "churn" in df.columns:
        return df["churn"].astype(str).str.strip().str.lower().eq("yes").astype(int)
    raise ValueError("Input data must contain 'churn_binary' or 'churn'.")


def drop_redundant_no_columns(columns: list[str]) -> list[str]:
    drop_cols: list[str] = []
    column_set = set(columns)
    for col in columns:
        if col.endswith("_no"):
            paired_yes = col[:-3] + "_yes"
            if paired_yes in column_set:
                drop_cols.append(col)
    return drop_cols


def to_v2_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """Convert input to v2-style 13-feature base table."""
    X = df.copy()

    for base_col in ["international_plan", "voice_mail_plan"]:
        if base_col in X.columns:
            X[base_col] = X[base_col].astype(str).str.strip().str.lower()
            X[f"{base_col}_yes"] = X[base_col].eq("yes").astype(int)
            X = X.drop(columns=[base_col])

    # Remove state and area_code families to match v2 base.
    if "state" in X.columns:
        X = X.drop(columns=["state"])
    state_cols = [col for col in X.columns if col.startswith("state_")]
    if state_cols:
        X = X.drop(columns=state_cols)

    if "area_code" in X.columns:
        X = X.drop(columns=["area_code"])
    area_code_cols = [col for col in X.columns if col.startswith("area_code_")]
    if area_code_cols:
        X = X.drop(columns=area_code_cols)

    no_cols = drop_redundant_no_columns(X.columns.tolist())
    if no_cols:
        X = X.drop(columns=no_cols)

    object_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if object_cols:
        for col in object_cols:
            X[col] = X[col].astype(str).str.strip().str.lower()
        X = pd.get_dummies(X, columns=object_cols, drop_first=False, dtype=int)

    bool_cols = X.select_dtypes(include=["bool"]).columns.tolist()
    for col in bool_cols:
        X[col] = X[col].astype(int)

    missing_features = [col for col in BASE_FEATURES_V2 if col not in X.columns]
    if missing_features:
        raise ValueError(f"Missing required base features for v3: {missing_features}")

    return X[BASE_FEATURES_V2].copy()


def engineer_features(base_df: pd.DataFrame) -> pd.DataFrame:
    """Create engineered features on top of v2 base features."""
    X = base_df.copy()

    # Aggregates
    X["total_minutes"] = (
        X["total_day_minutes"] + X["total_eve_minutes"] + X["total_night_minutes"] + X["total_intl_minutes"]
    )
    X["total_calls"] = (
        X["total_day_calls"] + X["total_eve_calls"] + X["total_night_calls"] + X["total_intl_calls"]
    )

    # Average duration per call
    X["avg_day_minutes_per_call"] = X["total_day_minutes"] / (X["total_day_calls"] + EPS)
    X["avg_eve_minutes_per_call"] = X["total_eve_minutes"] / (X["total_eve_calls"] + EPS)
    X["avg_night_minutes_per_call"] = X["total_night_minutes"] / (X["total_night_calls"] + EPS)
    X["avg_intl_minutes_per_call"] = X["total_intl_minutes"] / (X["total_intl_calls"] + EPS)

    # International usage ratios
    X["intl_minutes_ratio"] = X["total_intl_minutes"] / (X["total_minutes"] + EPS)
    X["intl_calls_ratio"] = X["total_intl_calls"] / (X["total_calls"] + EPS)

    # Interactions with international plan
    X["intl_plan_x_intl_minutes"] = X["international_plan_yes"] * X["total_intl_minutes"]
    X["intl_plan_x_intl_calls"] = X["international_plan_yes"] * X["total_intl_calls"]
    X["intl_plan_x_customer_service"] = X["international_plan_yes"] * X["number_customer_service_calls"]

    # Interactions with customer service burden
    X["cs_calls_x_day_minutes"] = X["number_customer_service_calls"] * X["total_day_minutes"]
    X["cs_calls_x_total_minutes"] = X["number_customer_service_calls"] * X["total_minutes"]

    # Service-call intensity flag
    X["high_customer_service_calls"] = (X["number_customer_service_calls"] >= 4).astype(int)

    # Voicemail usage behavior
    X["has_vmail_messages"] = (X["number_vmail_messages"] > 0).astype(int)
    X["vmail_plan_x_vmail_messages"] = X["voice_mail_plan_yes"] * X["number_vmail_messages"]

    return X


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    y = ensure_binary_target(df)
    raw_features = df.drop(columns=["churn_binary", "churn"], errors="ignore").copy()
    base_df = to_v2_base_features(raw_features)
    X = engineer_features(base_df)

    X = X.replace([np.inf, -np.inf], np.nan)
    if X.isna().any().any():
        raise ValueError("Found missing/invalid values after feature engineering.")

    feature_names = X.columns.tolist()
    return X, y, feature_names


def save_feature_importance_plot(importance_df: pd.DataFrame, path: Path, max_depth: int | None) -> None:
    top_df = importance_df.head(30).sort_values("importance", ascending=True)
    plt.figure(figsize=(10, max(5, len(top_df) * 0.27)))
    plt.barh(top_df["feature"], top_df["importance"], color="#E45756")
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.title(f"Model v3 RF feature importances (max_depth={max_depth})")
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def save_confusion_matrix_plot(
    y_true: pd.Series, y_pred: np.ndarray, path: Path, max_depth: int | None
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["no", "yes"])
    fig, ax = plt.subplots(figsize=(5, 4.5))
    disp.plot(ax=ax, cmap="Reds", values_format="d", colorbar=False)
    ax.set_title(f"Model v3 holdout confusion matrix (max_depth={max_depth})")
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_metrics_txt(metrics: dict[str, float | int | None], path: Path) -> None:
    lines = [
        "Phase 1 Model v3 Metrics (Feature Engineering)",
        "=" * 46,
        "",
    ]
    for key, value in metrics.items():
        if isinstance(value, float):
            lines.append(f"{key}: {value:.6f}")
        else:
            lines.append(f"{key}: {value}")
    path.write_text("\n".join(lines), encoding="utf-8")


def cv_confidence_interval(scores: np.ndarray, confidence: float = 0.95) -> tuple[float, float]:
    n = len(scores)
    if n < 2:
        value = float(scores.mean())
        return value, value
    mean_score = float(scores.mean())
    sem = stats.sem(scores)
    margin = float(sem * stats.t.ppf((1 + confidence) / 2, df=n - 1))
    return mean_score - margin, mean_score + margin


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.input_csv.exists():
        raise FileNotFoundError(f"Input file not found: {args.input_csv}")

    max_depth_values = parse_max_depths(args.max_depths)

    df = pd.read_csv(args.input_csv)
    X, y, feature_names = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_seed,
        stratify=y,
    )

    summary_rows: list[dict[str, float | int | None]] = []

    for max_depth in max_depth_values:
        model = RandomForestClassifier(max_depth=max_depth, random_state=args.random_seed)
        model.fit(X_train, y_train)

        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.random_seed)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1)
        cv_lower, cv_upper = cv_confidence_interval(cv_scores, confidence=0.95)

        max_depth_label = depth_tag(max_depth)
        metrics = {
            "n_samples_total": int(X.shape[0]),
            "n_features_used": int(X.shape[1]),
            "n_base_features": int(len(BASE_FEATURES_V2)),
            "n_engineered_features": int(X.shape[1] - len(BASE_FEATURES_V2)),
            "test_size": float(args.test_size),
            "random_seed": int(args.random_seed),
            "feature_strategy": "v2_base_plus_engineering",
            "max_depth": max_depth,
            "max_depth_label": max_depth_label,
            "train_f1": float(f1_score(y_train, train_pred, pos_label=POSITIVE_CLASS, zero_division=0)),
            "train_precision": float(
                precision_score(y_train, train_pred, pos_label=POSITIVE_CLASS, zero_division=0)
            ),
            "train_recall": float(recall_score(y_train, train_pred, pos_label=POSITIVE_CLASS, zero_division=0)),
            "holdout_f1": float(f1_score(y_test, test_pred, pos_label=POSITIVE_CLASS, zero_division=0)),
            "holdout_precision": float(
                precision_score(y_test, test_pred, pos_label=POSITIVE_CLASS, zero_division=0)
            ),
            "holdout_recall": float(recall_score(y_test, test_pred, pos_label=POSITIVE_CLASS, zero_division=0)),
            "holdout_accuracy": float(accuracy_score(y_test, test_pred)),
            "cv_f1_mean": float(cv_scores.mean()),
            "cv_f1_std": float(cv_scores.std(ddof=1)),
            "cv_f1_ci95_lower": float(cv_lower),
            "cv_f1_ci95_upper": float(cv_upper),
        }

        tag = depth_tag(max_depth)

        cv_df = pd.DataFrame({"fold": np.arange(1, len(cv_scores) + 1), "f1_score": cv_scores})
        cv_df.to_csv(args.output_dir / f"cv_f1_scores_v3_d{tag}.csv", index=False)

        importance_df = (
            pd.DataFrame({"feature": feature_names, "importance": model.feature_importances_})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
        importance_df.to_csv(args.output_dir / f"feature_importances_v3_d{tag}.csv", index=False)

        save_feature_importance_plot(importance_df, args.output_dir / f"feature_importance_v3_d{tag}.png", max_depth)
        save_confusion_matrix_plot(y_test, test_pred, args.output_dir / f"confusion_matrix_v3_d{tag}.png", max_depth)

        (args.output_dir / f"selected_features_v3_d{tag}.txt").write_text(
            "\n".join(feature_names), encoding="utf-8"
        )

        with open(args.output_dir / f"model_v3_d{tag}.pkl", "wb") as f:
            pickle.dump(
                {
                    "model": model,
                    "feature_names": feature_names,
                    "target_name": "churn_binary",
                    "random_seed": args.random_seed,
                    "max_depth": max_depth,
                    "feature_strategy": "v2_base_plus_engineering",
                },
                f,
            )

        with open(args.output_dir / f"metrics_v3_d{tag}.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        write_metrics_txt(metrics, args.output_dir / f"metrics_v3_d{tag}.txt")

        summary_rows.append(metrics)

        print("-" * 50)
        print(f"Model v3 complete (max_depth={max_depth}).")
        print(f"Features used: {len(feature_names)}")
        print(f"Train F1: {metrics['train_f1']:.6f}")
        print(f"Holdout F1: {metrics['holdout_f1']:.6f}")
        print(f"CV F1 mean (95% CI): {metrics['cv_f1_mean']:.6f} [{cv_lower:.6f}, {cv_upper:.6f}]")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(args.output_dir / "metrics_v3_by_max_depth.csv", index=False)
    print("Saved summary:", args.output_dir / "metrics_v3_by_max_depth.csv")


if __name__ == "__main__":
    main()
