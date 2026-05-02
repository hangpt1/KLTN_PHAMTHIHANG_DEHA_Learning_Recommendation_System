#!/usr/bin/env python3
"""
Evaluate non-RS ML components:
- Grade Predictor (RandomForestRegressor) with RMSE/MAE
- Student Clustering (K-Means) with Silhouette score
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

    for fold, (train_idx, test_idx) in enumerate(gkf.split(quizzes, groups=groups), start=1):
        train_quizzes = quizzes.iloc[train_idx].reset_index(drop=True)
        test_quizzes = quizzes.iloc[test_idx].reset_index(drop=True)

        predictor = GradePredictor()
        ok = predictor.fit(students, activities, train_quizzes, courses)
        if not ok:
            raise RuntimeError("GradePredictor could not be trained on the training fold.")

        X_test, y_test = predictor.prepare_features(students, activities, test_quizzes, courses)
        if X_test is None or len(X_test) == 0:
            continue

        y_pred = predictor.model.predict(predictor.scaler.transform(X_test))
        fold_metrics.append(
            FoldMetrics(
                rmse=rmse(y_pred, y_test),
                mae=mae(y_pred, y_test),
                n_test=int(len(y_test)),
            )
        )

        print(f"[Fold {fold}] n_test={len(y_test)} RMSE={fold_metrics[-1].rmse:.4f} MAE={fold_metrics[-1].mae:.4f}")

    if not fold_metrics:
        raise RuntimeError("No fold produced test metrics (unexpected empty feature set).")

    rmse_vals = np.array([m.rmse for m in fold_metrics], dtype=float)
    mae_vals = np.array([m.mae for m in fold_metrics], dtype=float)
    total_test = sum(m.n_test for m in fold_metrics)

    summary = {
        "folds": len(fold_metrics),
        "total_test_samples": int(total_test),
        "rmse_mean": float(rmse_vals.mean()),
        "rmse_std": float(rmse_vals.std(ddof=1)) if len(rmse_vals) > 1 else 0.0,
        "mae_mean": float(mae_vals.mean()),
        "mae_std": float(mae_vals.std(ddof=1)) if len(mae_vals) > 1 else 0.0,
    }
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


def main():
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, "data")

    print("\n" + "=" * 72)
    print("GRADE PREDICTOR EVALUATION (GroupKFold by student_id)")
    print("=" * 72)
    gp = evaluate_grade_predictor(data_dir=data_dir, n_splits=5)
    print(
        f"\nSummary: folds={gp['folds']}, total_test={gp['total_test_samples']}, "
        f"RMSE={gp['rmse_mean']:.4f}±{gp['rmse_std']:.4f}, "
        f"MAE={gp['mae_mean']:.4f}±{gp['mae_std']:.4f}"
    )

    print("\n" + "=" * 72)
    print("STUDENT CLUSTERING EVALUATION (Silhouette score)")
    print("=" * 72)
    cl = evaluate_clustering(data_dir=data_dir, n_clusters=4)
    print(f"Silhouette score: {cl['silhouette']:.4f}" if cl["silhouette"] is not None else "Silhouette score: N/A")
    print("Cluster distribution:")
    for cid, info in cl["distribution"].items():
        print(f"  - Cluster {cid}: {info['name']} ({info['count']} students, {info['percentage']}%)")


if __name__ == "__main__":
    main()
