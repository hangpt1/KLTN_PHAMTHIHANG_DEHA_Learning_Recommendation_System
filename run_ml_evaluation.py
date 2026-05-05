#!/usr/bin/env python3
"""
Đánh giá các thành phần ML không dựa trên RS:
- Dự đoán điểm số (RandomForestRegressor) với RMSE/MAE
- Phân cụm học sinh (K-Means) với điểm Silhouette
"""

import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Avoid noisy core-count warnings from joblib/loky on some macOS setups.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

from sklearn.model_selection import GroupKFold

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from ml_features import GradePredictor, StudentClusterer  # noqa: E402


@dataclass
class FoldMetrics:
    rmse: float
    mae: float
    n_test: int


def rmse(pred, actual) -> float:
    pred = np.asarray(pred, dtype=float)
    actual = np.asarray(actual, dtype=float)
    return float(np.sqrt(np.mean((pred - actual) ** 2)))


def mae(pred, actual) -> float:
    pred = np.asarray(pred, dtype=float)
    actual = np.asarray(actual, dtype=float)
    return float(np.mean(np.abs(pred - actual)))


def evaluate_grade_predictor(data_dir: str, n_splits: int = 5, random_state: int = 42):
    students = pd.read_csv(os.path.join(data_dir, "students.csv"))
    courses = pd.read_csv(os.path.join(data_dir, "courses.csv"))
    activities = pd.read_csv(os.path.join(data_dir, "activity_logs.csv"))
    quizzes = pd.read_csv(os.path.join(data_dir, "quiz_results.csv"))
    module_progress = pd.read_csv(os.path.join(data_dir, "module_progress.csv"))

    if quizzes.empty:
        raise RuntimeError("quiz_results.csv is empty; cannot evaluate GradePredictor.")

    # GroupKFold by student_id to reduce leakage across the same learner.
    groups = quizzes["student_id"].astype(str).values
    unique_groups = len(np.unique(groups))
    splits = min(n_splits, unique_groups) if unique_groups > 1 else 1
    if splits < 2:
        raise RuntimeError("Not enough unique student_id groups for cross-validation.")

    gkf = GroupKFold(n_splits=splits)
    fold_metrics: list[FoldMetrics] = []
    fold_metrics_in_progress: list[FoldMetrics] = []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(quizzes, groups=groups), start=1):
        train_quizzes = quizzes.iloc[train_idx].reset_index(drop=True)
        test_quizzes = quizzes.iloc[test_idx].reset_index(drop=True)

        predictor = GradePredictor()
        ok = predictor.fit(students, activities, train_quizzes, courses, module_progress_df=module_progress)
        if not ok:
            raise RuntimeError("GradePredictor could not be trained on the training fold.")

        # Evaluate pre_start mode (no course-specific activity features)
        X_test, y_test = predictor.prepare_features(
            students, activities, test_quizzes, courses, module_progress_df=module_progress, mode="pre_start"
        )
        if X_test is not None and len(X_test) > 0:
            y_pred = predictor.model_pre_start.predict(predictor.scaler_pre_start.transform(X_test))
            fold_metrics.append(
                FoldMetrics(
                    rmse=rmse(y_pred, y_test),
                    mae=mae(y_pred, y_test),
                    n_test=int(len(y_test)),
                )
            )
            print(
                f"[Fold {fold} | pre_start] n_test={len(y_test)} "
                f"RMSE={fold_metrics[-1].rmse:.4f} MAE={fold_metrics[-1].mae:.4f}"
            )

        # Evaluate in_progress mode (includes activity features when available)
        X_test_in, y_test_in = predictor.prepare_features(
            students, activities, test_quizzes, courses, module_progress_df=module_progress, mode="in_progress"
        )
        if X_test_in is not None and len(X_test_in) > 0:
            y_pred_in = predictor.model_in_progress.predict(predictor.scaler_in_progress.transform(X_test_in))
            fold_metrics_in_progress.append(
                FoldMetrics(
                    rmse=rmse(y_pred_in, y_test_in),
                    mae=mae(y_pred_in, y_test_in),
                    n_test=int(len(y_test_in)),
                )
            )
            print(
                f"[Fold {fold} | in_progress] n_test={len(y_test_in)} "
                f"RMSE={fold_metrics_in_progress[-1].rmse:.4f} MAE={fold_metrics_in_progress[-1].mae:.4f}"
            )

    if not fold_metrics and not fold_metrics_in_progress:
        raise RuntimeError("No fold produced test metrics (unexpected empty feature set).")

    summary = {"folds": int(splits)}

    if fold_metrics:
        rmse_vals = np.array([m.rmse for m in fold_metrics], dtype=float)
        mae_vals = np.array([m.mae for m in fold_metrics], dtype=float)
        total_test = sum(m.n_test for m in fold_metrics)
        summary.update(
            {
                "pre_start_total_test_samples": int(total_test),
                "pre_start_rmse_mean": float(rmse_vals.mean()),
                "pre_start_rmse_std": float(rmse_vals.std(ddof=1)) if len(rmse_vals) > 1 else 0.0,
                "pre_start_mae_mean": float(mae_vals.mean()),
                "pre_start_mae_std": float(mae_vals.std(ddof=1)) if len(mae_vals) > 1 else 0.0,
            }
        )

    if fold_metrics_in_progress:
        rmse_vals = np.array([m.rmse for m in fold_metrics_in_progress], dtype=float)
        mae_vals = np.array([m.mae for m in fold_metrics_in_progress], dtype=float)
        total_test = sum(m.n_test for m in fold_metrics_in_progress)
        summary.update(
            {
                "in_progress_total_test_samples": int(total_test),
                "in_progress_rmse_mean": float(rmse_vals.mean()),
                "in_progress_rmse_std": float(rmse_vals.std(ddof=1)) if len(rmse_vals) > 1 else 0.0,
                "in_progress_mae_mean": float(mae_vals.mean()),
                "in_progress_mae_std": float(mae_vals.std(ddof=1)) if len(mae_vals) > 1 else 0.0,
            }
        )
    return summary


def evaluate_clustering(data_dir: str, n_clusters: int = 4):
    students = pd.read_csv(os.path.join(data_dir, "students.csv"))
    activities = pd.read_csv(os.path.join(data_dir, "activity_logs.csv"))
    quizzes = pd.read_csv(os.path.join(data_dir, "quiz_results.csv"))

    clusterer = StudentClusterer(n_clusters=n_clusters)
    ok = clusterer.fit(students, activities, quizzes)
    if not ok:
        raise RuntimeError("StudentClusterer could not be fitted.")

    silhouette = clusterer.silhouette(students, activities, quizzes)
    distribution = clusterer.get_all_clusters(students, activities, quizzes)
    return {"silhouette": silhouette, "distribution": distribution}


def _save_ml_component_report(
    report_path: str,
    gp: dict,
    cl: dict,
    n_clusters: int,
) -> None:
    """Append a compact text snapshot for thesis / bookkeeping."""
    lines = []
    lines.append("ML COMPONENTS EVALUATION SNAPSHOT")
    lines.append("=" * 72)
    lines.append("")
    lines.append("GRADE PREDICTOR (GroupKFold by student_id, n_splits=5)")
    lines.append("-" * 72)
    for k in sorted(gp.keys()):
        lines.append(f"  {k}: {gp[k]}")
    lines.append("")
    lines.append("STUDENT CLUSTERING (K-Means)")
    lines.append("-" * 72)
    lines.append(f"  silhouette_score: {cl.get('silhouette')}")
    lines.append(f"  n_clusters_requested: {n_clusters}")
    if cl.get("distribution"):
        lines.append("  distribution:")
        for cid, info in cl["distribution"].items():
            lines.append(
                f"    Cluster {cid}: {info['name']} "
                f"({info['count']} students, {info['percentage']}%)"
            )
    text = "\n".join(lines) + "\n"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(text)


def main():
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, "data")

    print("\n" + "=" * 72)
    print("GRADE PREDICTOR EVALUATION (GroupKFold by student_id)")
    print("=" * 72)
    gp = evaluate_grade_predictor(data_dir=data_dir, n_splits=5)
    if "pre_start_rmse_mean" in gp:
        print(
            f"\nSummary (pre_start): folds={gp['folds']}, total_test={gp['pre_start_total_test_samples']}, "
            f"RMSE={gp['pre_start_rmse_mean']:.4f}±{gp['pre_start_rmse_std']:.4f}, "
            f"MAE={gp['pre_start_mae_mean']:.4f}±{gp['pre_start_mae_std']:.4f}"
        )
    if "in_progress_rmse_mean" in gp:
        print(
            f"\nSummary (in_progress): folds={gp['folds']}, total_test={gp['in_progress_total_test_samples']}, "
            f"RMSE={gp['in_progress_rmse_mean']:.4f}±{gp['in_progress_rmse_std']:.4f}, "
            f"MAE={gp['in_progress_mae_mean']:.4f}±{gp['in_progress_mae_std']:.4f}"
        )

    print("\n" + "=" * 72)
    print("STUDENT CLUSTERING EVALUATION (Silhouette score)")
    print("=" * 72)
    n_clusters = 4
    cl = evaluate_clustering(data_dir=data_dir, n_clusters=n_clusters)
    print(f"Silhouette score: {cl['silhouette']:.4f}" if cl["silhouette"] is not None else "Silhouette score: N/A")
    print("Cluster distribution:")
    for cid, info in cl["distribution"].items():
        print(f"  - Cluster {cid}: {info['name']} ({info['count']} students, {info['percentage']}%)")

    report_path = os.path.join(base_dir, "evaluation_results", "ml_components_report.txt")
    _save_ml_component_report(report_path, gp, cl, n_clusters=n_clusters)
    print(f"\n✓ ML summary written to: {report_path}")


if __name__ == "__main__":
    main()
