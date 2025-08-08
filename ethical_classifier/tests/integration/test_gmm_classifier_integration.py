"""
Integration Test for GMMClassifier with Synthetic Data

Generates synthetic VDC vectors (Virtue, Deontological, Consequentialist)
with five separable clusters (BLUE..RED) and optional intent weights.
Validates training and prediction accuracy to ensure the reconstructed
classifier behaves as expected in a non-integrated pipeline.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split

from ethical_classifier.core import GMMClassifier, SeverityLevel


def _generate_synthetic_vdc(n_samples: int = 1000, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    # Cluster centers for each severity (0..4)
    centers = {
        0: np.array([0.10, 0.10, 0.10]),  # BLUE
        1: np.array([0.30, 0.20, 0.20]),  # GREEN
        2: np.array([0.50, 0.50, 0.50]),  # YELLOW
        3: np.array([0.70, 0.60, 0.60]),  # ORANGE
        4: np.array([0.90, 0.80, 0.80]),  # RED
    }
    per_class = n_samples // len(centers)
    X_list, y_list = [], []
    for label, c in centers.items():
        # Use a constant small std for stronger separability in integration test
        std = 0.05
        cov = np.diag([std**2, std**2, std**2])
        pts = rng.multivariate_normal(c, cov, size=per_class)
        pts = np.clip(pts, 0.0, 1.0)
        X_list.append(pts)
        y_list.extend([label] * per_class)
    X = np.vstack(X_list)
    y = np.asarray(y_list)
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def _generate_pipeline_intent_weights(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Simulate pipeline-grade intent by deriving a harm-intensity scalar from labels.
    Higher severity -> higher intent intensity; then convert to per-dimension weights.
    """
    # Harm intensity in [0,1]
    h = (y.astype(float) - float(SeverityLevel.BLUE)) / (float(SeverityLevel.RED) - float(SeverityLevel.BLUE))
    h = h.reshape(-1, 1)
    # Convert scalar intensity to per-dimension weights that amplify deviation from neutral 0.5
    beta = 1.2
    W = 1.0 + beta * h * (X - 0.5)
    return np.clip(W, 0.6, 2.0)


def test_gmm_classifier_synthetic_high_accuracy(tmp_path: Path) -> None:
    # Data
    X, y = _generate_synthetic_vdc(n_samples=2000, seed=123)
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123, stratify=y
    )
    # Pipeline-grade intent simulation
    W_train = _generate_pipeline_intent_weights(X_train, y_train)
    W_test = _generate_pipeline_intent_weights(X_test, y_test)

    # Classifier with spectrum-density aligner explicitly enabled
    clf = GMMClassifier(
        n_components=5,
        random_state=123,
        covariance_type="diag",
        max_iter=300,
        n_init=5,
        verbose=0,
        mag_weight=1.0,
        use_spectrum_aligner=True,
        aligner_params={
            "density_algorithm": "dbscan",  # deterministic fallback
            "eps": 0.3,
            "min_samples": 5,
            "n_spectral_clusters": 5,
            "affinity": "rbf",
            "gamma": 1.0,
            "n_neighbors": 10,
            "random_state": 123,
        },
    )

    # Supervised mapping: provide labels during fit for high separability
    clf.fit(X_train, y=y_train, intent_weights=W_train)

    # Predict
    y_pred = clf.predict(X_test, intent_weights=W_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    # Report for debugging (pytest -s to show)
    report: str = classification_report(y_test, y_pred, digits=4)
    print("\nGMM synthetic classification report:\n", report)
    print(f"Accuracy: {acc:.4f}, F1: {f1:.4f}")

    # Expect high accuracy on separable synthetic data
    assert acc >= 0.90, f"Expected >=0.90 accuracy, got {acc:.4f}"
    assert f1 >= 0.90, f"Expected >=0.90 F1, got {f1:.4f}"

    # Sanity: names API
    names = clf.predict(X_test[:5], return_severity_name=True)
    assert isinstance(names[0], str)
