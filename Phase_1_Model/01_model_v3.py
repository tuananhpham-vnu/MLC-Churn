"""Phase 1 model v2: Random Forest without state features.

Policy:
- target: churn_binary (or derive from churn if needed)
- drop all state-derived features (state, state_*)
- drop all area_code-derived features
- keep binary yes indicators and drop redundant *_no columns
- run multiple max_depth values (default: 3, 5, None)

Usage:
    python Phase_1_Model/01_model_v2.py
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


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


# ===== KHU VUC CONFIG =====
ROOT_DIR = project_root()
DEFAULT_INPUT = ROOT_DIR / "Phase_0_EDA" / "outputs" / "preprocessed_improved.csv"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "Phase_1_Model" / "outputs"
DEFAULT_MAX_DEPTHS = "3,5,None"



def parse_args() -> argparse.Namespace:
    # [CONFIG] Tham so chay script
    parser = argparse.ArgumentParser(description="Train RF model v2 (without state features).")
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


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    # [CONFIG/PREPROCESS] Chuan hoa feature theo policy cua model v2
    y = ensure_binary_target(df)
    X = df.drop(columns=["churn_binary", "churn"], errors="ignore").copy()

    # Neu con cot raw nhi phan, doi sang *_yes cho ro rang.
    for base_col in ["international_plan", "voice_mail_plan"]:
        if base_col in X.columns:
            X[base_col] = X[base_col].astype(str).str.strip().str.lower()
            X[f"{base_col}_yes"] = X[base_col].eq("yes").astype(int)
            X = X.drop(columns=[base_col])

    # Loai bo state ngay ca khi du lieu o dang raw hoac one-hot.
    if "state" in X.columns:
        X = X.drop(columns=["state"])
    state_cols = [col for col in X.columns if col.startswith("state_")]
    if state_cols:
        X = X.drop(columns=state_cols)

    # Loai bo area_code raw + one-hot.
    if "area_code" in X.columns:
        X = X.drop(columns=["area_code"])
    area_code_cols = [col for col in X.columns if col.startswith("area_code_")]
    if area_code_cols:
        X = X.drop(columns=area_code_cols)

    # Drop redundant *_no columns when *_yes exists.
    no_cols = drop_redundant_no_columns(X.columns.tolist())
    if no_cols:
        X = X.drop(columns=no_cols)

    # Safety: encode categorical con lai (neu co).
    object_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if object_cols:
        for col in object_cols:
            X[col] = X[col].astype(str).str.strip().str.lower()
        X = pd.get_dummies(X, columns=object_cols, drop_first=False, dtype=int)

    bool_cols = X.select_dtypes(include=["bool"]).columns.tolist()
    for col in bool_cols:
        X[col] = X[col].astype(int)

    X = X.replace([np.inf, -np.inf], np.nan)
    if X.isna().any().any():
        raise ValueError("Found missing/invalid values in feature matrix after preprocessing.")

    selected_features = X.columns.tolist()
    return X, y, selected_features


def save_feature_importance_plot(importance_df: pd.DataFrame, path: Path, max_depth: int | None) -> None:
    # [PLOT] Ve bieu do do quan trong feature
    top_df = importance_df.head(25).sort_values("importance", ascending=True)
    plt.figure(figsize=(9, max(5, len(top_df) * 0.28)))
    plt.barh(top_df["feature"], top_df["importance"], color="#B279A2")
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.title(f"Model v2 RF feature importances (max_depth={max_depth})")
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def save_confusion_matrix_plot(
    y_true: pd.Series, y_pred: np.ndarray, path: Path, max_depth: int | None
) -> None:
    # [PLOT] Ve confusion matrix tren tap holdout
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["no", "yes"])
    fig, ax = plt.subplots(figsize=(5, 4.5))
    disp.plot(ax=ax, cmap="Purples", values_format="d", colorbar=False)
    ax.set_title(f"Model v2 holdout confusion matrix (max_depth={max_depth})")
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_metrics_txt(metrics: dict[str, float | int | None], path: Path) -> None:
    lines = [
        "Phase 1 Model v2 Metrics (Without State)",
        "=" * 42,
        "",
    ]
    for key, value in metrics.items():
        if isinstance(value, float):
            lines.append(f"{key}: {value:.6f}")
        else:
            lines.append(f"{key}: {value}")
    path.write_text("\n".join(lines), encoding="utf-8")


def cv_confidence_interval(scores: np.ndarray, confidence: float = 0.95) -> tuple[float, float]:
    # [EVAL] Tinh khoang tin cay cho CV F1
    n = len(scores)
    if n < 2:
        value = float(scores.mean())
        return value, value
    mean_score = float(scores.mean())
    sem = stats.sem(scores)
    margin = float(sem * stats.t.ppf((1 + confidence) / 2, df=n - 1))
    return mean_score - margin, mean_score + margin


def main() -> None:
    # ===== [CONFIG] Doc tham so + tao thu muc output =====
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.input_csv.exists():
        raise FileNotFoundError(f"Input file not found: {args.input_csv}")

    max_depth_values = parse_max_depths(args.max_depths)

    # ===== [CONFIG/PREPROCESS] Doc du lieu + tao ma tran feature =====
    df = pd.read_csv(args.input_csv)
    X, y, selected_features = prepare_features(df)

    # ===== [TRAIN] Chia tap train/holdout theo stratified =====
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_seed,
        stratify=y,
    )

    summary_rows: list[dict[str, float | int | None]] = []

    for max_depth in max_depth_values:
        # ===== [TRAIN] Khoi tao va huan luyen Random Forest =====
        model = RandomForestClassifier(
            max_depth=max_depth,
            random_state=args.random_seed,
        )
        model.fit(X_train, y_train)

        # ===== [EVAL] Du doan va tinh metric =====
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.random_seed)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1)
        cv_lower, cv_upper = cv_confidence_interval(cv_scores, confidence=0.95)

        max_depth_label = depth_tag(max_depth)
        metrics = {
            "n_samples_total": int(X.shape[0]),
            "n_features_used": int(X.shape[1]),
            "test_size": float(args.test_size),
            "random_seed": int(args.random_seed),
            "state_policy": "drop_all_state_features",
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

        # ===== [SAVE] Luu diem CV va bang importance =====
        cv_df = pd.DataFrame({"fold": np.arange(1, len(cv_scores) + 1), "f1_score": cv_scores})
        cv_df.to_csv(args.output_dir / f"cv_f1_scores_v2_d{tag}.csv", index=False)

        importance_df = (
            pd.DataFrame({"feature": selected_features, "importance": model.feature_importances_})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
        importance_df.to_csv(args.output_dir / f"feature_importances_v2_d{tag}.csv", index=False)

        # ===== [PLOT] Ve cac hinh =====
        save_feature_importance_plot(importance_df, args.output_dir / f"feature_importance_v2_d{tag}.png", max_depth)
        save_confusion_matrix_plot(y_test, test_pred, args.output_dir / f"confusion_matrix_v2_d{tag}.png", max_depth)

        # ===== [SAVE] Luu feature list, model va metrics =====
        (args.output_dir / f"selected_features_v2_d{tag}.txt").write_text(
            "\n".join(selected_features), encoding="utf-8"
        )

        with open(args.output_dir / f"model_v2_d{tag}.pkl", "wb") as f:
            pickle.dump(
                {
                    "model": model,
                    "feature_names": selected_features,
                    "target_name": "churn_binary",
                    "random_seed": args.random_seed,
                    "max_depth": max_depth,
                    "state_policy": "drop_all_state_features",
                },
                f,
            )

        with open(args.output_dir / f"metrics_v2_d{tag}.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        write_metrics_txt(metrics, args.output_dir / f"metrics_v2_d{tag}.txt")

        summary_rows.append(metrics)

        print("-" * 50)
        print(f"Model v2 complete (max_depth={max_depth}).")
        print(f"Features used: {len(selected_features)}")
        print(f"Train F1: {metrics['train_f1']:.6f}")
        print(f"Holdout F1: {metrics['holdout_f1']:.6f}")
        print(f"CV F1 mean (95% CI): {metrics['cv_f1_mean']:.6f} [{cv_lower:.6f}, {cv_upper:.6f}]")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(args.output_dir / "metrics_v2_by_max_depth.csv", index=False)
    print("Saved summary:", args.output_dir / "metrics_v2_by_max_depth.csv")


if __name__ == "__main__":
    main()
