"""Phase 1 baseline Random Forest model for MLC churn."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import matplotlib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TARGET = "churn_binary"
AREA_PREFIX = "area_code_"
CHARGE_SUFFIX = "_charge"
ROOT_DIR = Path(__file__).resolve().parents[1]


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    return (ROOT_DIR / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Phase 1 baseline Random Forest model.")
    parser.add_argument(
        "--input-csv",
        default="Phase_0_EDA/outputs/preprocessed_improved.csv",
        help="Path to preprocessed CSV with churn_binary target.",
    )
    parser.add_argument(
        "--output-dir",
        default="Phase_1_Model/outputs",
        help="Directory to save model artifacts.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=1234,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")
    df = pd.read_csv(path)
    if TARGET not in df.columns:
        raise ValueError(f"Required target column '{TARGET}' not found in {path}.")
    return df


def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    area_cols = [column for column in df.columns if column.startswith(AREA_PREFIX)]
    charge_cols = [column for column in df.columns if column.endswith(CHARGE_SUFFIX)]
    drop_cols = sorted(set(area_cols + charge_cols + [TARGET]))

    feature_cols = [column for column in df.columns if column not in drop_cols]
    if not feature_cols:
        raise ValueError("No feature columns remain after applying drop policy.")

    x = df[feature_cols].copy()
    y = df[TARGET].astype(int).copy()
    return x, y


def save_confusion_matrix(y_true: pd.Series, y_pred: pd.Series, output_path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["no(0)", "yes(1)"])
    display.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Baseline confusion matrix (train set)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_feature_importance(model: RandomForestClassifier, feature_names: list[str], csv_path: Path, png_path: Path) -> None:
    importance_df = pd.DataFrame({"feature": feature_names, "importance": model.feature_importances_})
    importance_df = importance_df.sort_values("importance", ascending=False).reset_index(drop=True)
    importance_df.to_csv(csv_path, index=False)

    top_df = importance_df.head(20).sort_values("importance", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(top_df["feature"], top_df["importance"], color="#4C78A8")
    ax.set_xlabel("Importance")
    ax.set_title("Top 20 feature importances - baseline")
    fig.tight_layout()
    fig.savefig(png_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_path = resolve_path(args.input_csv)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(input_path)
    x, y = build_feature_matrix(df)

    model = RandomForestClassifier(random_state=args.random_seed, max_depth=None, n_jobs=-1)
    model.fit(x, y)
    train_pred = model.predict(x)

    train_f1 = float(f1_score(y, train_pred, pos_label=1))
    train_precision = float(precision_score(y, train_pred, pos_label=1, zero_division=0))
    train_recall = float(recall_score(y, train_pred, pos_label=1, zero_division=0))

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.random_seed)
    cv_scores = cross_val_score(model, x, y, cv=cv, scoring="f1", n_jobs=-1)
    cv_df = pd.DataFrame({"fold": range(1, len(cv_scores) + 1), "f1": cv_scores})
    cv_path = output_dir / "cv_f1_scores_baseline.csv"
    cv_df.to_csv(cv_path, index=False)

    metrics = {
        "dataset_rows": int(df.shape[0]),
        "dataset_columns": int(df.shape[1]),
        "n_features_used": int(x.shape[1]),
        "train_f1": train_f1,
        "train_precision": train_precision,
        "train_recall": train_recall,
        "cv_f1_mean": float(cv_scores.mean()),
        "cv_f1_std": float(cv_scores.std(ddof=1)),
        "cv_f1_min": float(cv_scores.min()),
        "cv_f1_max": float(cv_scores.max()),
        "max_depth": None,
        "random_state": args.random_seed,
    }

    metrics_json_path = output_dir / "metrics_baseline.json"
    metrics_json_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    metrics_txt_lines = [
        "PHASE 1 BASELINE RANDOM FOREST METRICS",
        "=" * 42,
        f"Input CSV: {input_path}",
        f"Rows x columns: {df.shape[0]} x {df.shape[1]}",
        f"Features used: {x.shape[1]}",
        "",
        f"Train F1: {train_f1:.6f}",
        f"Train Precision: {train_precision:.6f}",
        f"Train Recall: {train_recall:.6f}",
        "",
        f"CV F1 mean: {cv_scores.mean():.6f}",
        f"CV F1 std: {cv_scores.std(ddof=1):.6f}",
        f"CV F1 min: {cv_scores.min():.6f}",
        f"CV F1 max: {cv_scores.max():.6f}",
        "",
        f"Model random_state: {args.random_seed}",
        "Model max_depth: None",
    ]
    (output_dir / "metrics_baseline.txt").write_text("\n".join(metrics_txt_lines), encoding="utf-8")

    with (output_dir / "model_baseline.pkl").open("wb") as f:
        pickle.dump(model, f)

    (output_dir / "selected_features_baseline.txt").write_text("\n".join(x.columns.tolist()), encoding="utf-8")

    save_feature_importance(
        model=model,
        feature_names=x.columns.tolist(),
        csv_path=output_dir / "feature_importances_baseline.csv",
        png_path=output_dir / "feature_importance_baseline.png",
    )
    save_confusion_matrix(y_true=y, y_pred=train_pred, output_path=output_dir / "confusion_matrix_baseline.png")

    print("Phase 1 baseline completed.")
    print(f"Input: {input_path}")
    print(f"Output directory: {output_dir}")
    print(f"Features used: {x.shape[1]}")
    print(f"Train F1: {train_f1:.6f}")
    print(f"CV F1 mean +/- std: {cv_scores.mean():.6f} +/- {cv_scores.std(ddof=1):.6f}")


if __name__ == "__main__":
    main()
