"""
VoxTriad – UCI Oxford Parkinson's Dataset Training Script
Phase 3 Backend

Trains the EvidentialMetaEnsemble on the clinical UCI dataset:
  - 195 acoustic recordings
  - 31 unique subjects
  - 22 biomedical voice measurement features
  - Leave-One-Subject-Out Cross-Validation (LOSO-CV) to prevent patient leakage
  - Serialises fitted model to backend/ensemble_model.joblib
"""

import os
import pathlib
import sys
import numpy as np
import pandas as pd

# Ensure backend directory is in sys.path
BASE_DIR = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from ensemble import EvidentialMetaEnsemble, DEFAULT_MODEL_PATH
from feature_extraction import UCI_FEATURE_NAMES


def train_uci(
    csv_path: str = None,
    save_path: str = None,
    verbose: bool = True,
) -> EvidentialMetaEnsemble:
    """
    Load UCI Parkinson's dataset, perform LOSO-CV, refit on full data,
    and persist the resulting ensemble artifact.
    """
    if csv_path is None:
        csv_path = PROJECT_ROOT / "data" / "parkinsons.csv"
        if not csv_path.exists():
            csv_path = BASE_DIR.parent / "data" / "parkinsons.csv"

    csv_path = pathlib.Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"UCI dataset not found at: {csv_path}")

    print(f"Loading UCI Parkinson's Dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Dataset shape: {df.shape}")

    # Extract labels (status: 0 = Healthy, 1 = Parkinson's)
    y = df["status"].values.astype(np.int32)

    # Subject grouping from name (e.g. 'phon_R01_S01_1' -> 'phon_R01_S01')
    groups = df["name"].apply(lambda x: "_".join(x.split("_")[:3])).values
    unique_subjects = np.unique(groups)
    print(f"Total samples: {len(df)} | Unique subjects: {len(unique_subjects)}")
    print(f"Class distribution: Healthy (0) = {(y == 0).sum()}, PD (1) = {(y == 1).sum()}")

    # Feature matrix: 22 UCI features in canonical order
    missing_cols = [col for col in UCI_FEATURE_NAMES if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected UCI columns in CSV: {missing_cols}")

    X = df[UCI_FEATURE_NAMES].values.astype(np.float32)
    print(f"Feature matrix shape: {X.shape} (22 features)")

    # Initialize and train ensemble with LOSO-CV
    ensemble = EvidentialMetaEnsemble()
    metrics = ensemble.train_with_loso(X, y, groups, verbose=verbose)

    print("\n" + "=" * 65)
    print("UCI LOSO-CV Evaluation Summary")
    print("=" * 65)
    print(f"  Folds (Subjects):  {metrics['n_folds']}")
    print(f"  Mean Accuracy:     {metrics['mean_accuracy'] * 100:.2f}%")
    print(f"  Mean F1-Score:     {metrics['mean_f1'] * 100:.2f}%")
    if metrics["mean_auc"] is not None and not np.isnan(metrics["mean_auc"]):
        print(f"  Mean ROC-AUC:      {metrics['mean_auc'] * 100:.2f}%")
    print("=" * 65)

    # Save trained ensemble artifact
    out_path = save_path or DEFAULT_MODEL_PATH
    ensemble.save(out_path)
    print(f"[SUCCESS] Trained UCI ensemble successfully persisted to {out_path}")

    return ensemble


if __name__ == "__main__":
    train_uci()
