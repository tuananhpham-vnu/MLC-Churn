"""Phase 1 improved Random Forest model for MLC churn."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import matplotlib
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

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
    parser = argparse.ArgumentParser(description="Train Phase 1 improved Random Forest model.")
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


def mean_ci95(values: list[float]) -> tuple[float, float, float]:
    series = pd.Series(values, dtype=float)
    mean = float(series.mean())
    if len(series) < 2:
        return mean, mean, mean
    sem = float(series.sem(ddof=1))
    t_crit = float(stats.t.ppf(0.975, df=len(series) - 1))
    margin = t_crit * sem
    return mean, mean - margin, mean + margin


def save_feature_importance(model: RandomForestClassifier, feature_names: list[str], csv_path: Path, png_path: Path) -> None:
    raw_importance = pd.Series(model.feature_importances_, index=feature_names)
    norm_importance = raw_importance / raw_importance.sum() if raw_importance.sum() > 0 else raw_importance

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": raw_importance.values,
            "normalized_importance": norm_importance.values,
        }
    )
    importance_df = importance_df.sort_values("normalized_importance", ascending=False).reset_index(drop=True)
    importance_df.to_csv(csv_path, index=False)

    plot_df = importance_df.head(20).sort_values("normalized_importance", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(plot_df["feature"], plot_df["normalized_importance"], color="#54A24B")
    ax.set_xlabel("Normalized importance")
    ax.set_title("Top 20 normalized feature importances - improved")
    fig.tight_layout()
    fig.savefig(png_path, dpi=160)
    plt.close(fig)


def save_confusion_matrix(y_true: pd.Series, y_pred: pd.Series, output_path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["no(0)", "yes(1)"])
    display.plot(ax=ax, cmap="Oranges", colorbar=False)
    ax.set_title("Improved confusion matrix (holdout)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_roc_pr_curves(y_true: pd.Series, y_score: pd.Series, roc_path: Path, pr_path: Path) -> tuple[float, float]:
    roc_auc = float(roc_auc_score(y_true, y_score))
    ap = float(average_precision_score(y_true, y_score))

    fig1, ax1 = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_true, y_score, ax=ax1, name=f"ROC AUC={roc_auc:.3f}")
    ax1.set_title("Improved ROC curve (holdout)")
    fig1.tight_layout()
    fig1.savefig(roc_path, dpi=160)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_predictions(y_true, y_score, ax=ax2, name=f"AP={ap:.3f}")
    ax2.set_title("Improved Precision-Recall curve (holdout)")
    fig2.tight_layout()
    fig2.savefig(pr_path, dpi=160)
    plt.close(fig2)

    return roc_auc, ap


def main() -> None:
    args = parse_args()
    input_path = resolve_path(args.input_csv)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(input_path)
    x, y = build_feature_matrix(df)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=args.random_seed
    )

    model = RandomForestClassifier(random_state=args.random_seed, max_depth=None, n_jobs=-1)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    y_score = model.predict_proba(x_test)[:, 1]

    holdout_metrics = {
        "holdout_accuracy": float(accuracy_score(y_test, y_pred)),
        "holdout_f1": float(f1_score(y_test, y_pred, pos_label=1)),
        "holdout_precision": float(precision_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "holdout_recall": float(recall_score(y_test, y_pred, pos_label=1, zero_division=0)),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.random_seed)
    cv_scores = cross_val_score(model, x, y, cv=cv, scoring="f1", n_jobs=-1)
    cv_df = pd.DataFrame({"fold": range(1, len(cv_scores) + 1), "f1": cv_scores})
    cv_df.to_csv(output_dir / "cv_f1_scores_improved.csv", index=False)

    cv_mean, cv_ci_low, cv_ci_high = mean_ci95(cv_scores.tolist())

    roc_auc, ap = save_roc_pr_curves(
        y_true=y_test,
        y_score=y_score,
        roc_path=output_dir / "roc_curve_improved.png",
        pr_path=output_dir / "pr_curve_improved.png",
    )

    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    report_txt = classification_report(y_test, y_pred, zero_division=0)
    (output_dir / "classification_report_improved.json").write_text(
        json.dumps(report_dict, indent=2), encoding="utf-8"
    )
    (output_dir / "classification_report_improved.txt").write_text(report_txt, encoding="utf-8")

    save_confusion_matrix(y_true=y_test, y_pred=y_pred, output_path=output_dir / "confusion_matrix_improved.png")

    save_feature_importance(
        model=model,
        feature_names=x.columns.tolist(),
        csv_path=output_dir / "feature_importances_improved.csv",
        png_path=output_dir / "feature_importance_improved.png",
    )

    with (output_dir / "model_improved.pkl").open("wb") as f:
        pickle.dump(model, f)

    (output_dir / "selected_features_improved.txt").write_text("\n".join(x.columns.tolist()), encoding="utf-8")

    metrics = {
        "dataset_rows": int(df.shape[0]),
        "dataset_columns": int(df.shape[1]),
        "n_features_used": int(x.shape[1]),
        "train_rows": int(x_train.shape[0]),
        "test_rows": int(x_test.shape[0]),
        "holdout": holdout_metrics,
        "cv_f1_mean": float(cv_mean),
        "cv_f1_std": float(cv_scores.std(ddof=1)),
        "cv_f1_min": float(cv_scores.min()),
        "cv_f1_max": float(cv_scores.max()),
        "cv_f1_ci95_low": float(cv_ci_low),
        "cv_f1_ci95_high": float(cv_ci_high),
        "roc_auc_holdout": float(roc_auc),
        "average_precision_holdout": float(ap),
        "max_depth": None,
        "random_state": args.random_seed,
    }
    (output_dir / "metrics_improved.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    metrics_txt_lines = [
        "PHASE 1 IMPROVED RANDOM FOREST METRICS",
        "=" * 42,
        f"Input CSV: {input_path}",
        f"Rows x columns: {df.shape[0]} x {df.shape[1]}",
        f"Features used: {x.shape[1]}",
        f"Train/Test rows: {x_train.shape[0]}/{x_test.shape[0]}",
        "",
        f"Holdout Accuracy: {holdout_metrics['holdout_accuracy']:.6f}",
        f"Holdout F1: {holdout_metrics['holdout_f1']:.6f}",
        f"Holdout Precision: {holdout_metrics['holdout_precision']:.6f}",
        f"Holdout Recall: {holdout_metrics['holdout_recall']:.6f}",
        f"ROC AUC (holdout): {roc_auc:.6f}",
        f"Average Precision (holdout): {ap:.6f}",
        "",
        f"CV F1 mean: {cv_mean:.6f}",
        f"CV F1 std: {cv_scores.std(ddof=1):.6f}",
        f"CV F1 95% CI: [{cv_ci_low:.6f}, {cv_ci_high:.6f}]",
        "",
        f"Model random_state: {args.random_seed}",
        "Model max_depth: None",
    ]
    (output_dir / "metrics_improved.txt").write_text("\n".join(metrics_txt_lines), encoding="utf-8")

    print("Phase 1 improved completed.")
    print(f"Input: {input_path}")
    print(f"Output directory: {output_dir}")
    print(f"Features used: {x.shape[1]}")
    print(f"Holdout F1: {holdout_metrics['holdout_f1']:.6f}")
    print(f"CV F1 mean (95% CI): {cv_mean:.6f} [{cv_ci_low:.6f}, {cv_ci_high:.6f}]")


if __name__ == "__main__":
    main()
