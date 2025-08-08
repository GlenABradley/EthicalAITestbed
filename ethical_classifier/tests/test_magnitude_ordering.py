import numpy as np
import math
import pytest

from ethical_classifier.core.gmm_classifier import GMMClassifier, SeverityLevel


def compute_expected_mag(vdc: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Reference implementation of worst-axis * intent magnitude.
    vdc: (n,3) raw V/D/C
    W:   (n,3) intent weights aligned to V/D/C
    returns: (n,1) mag = log1p((1 - min(V,D,C)) * intent_on_worst)
    """
    n = vdc.shape[0]
    worst_idx = np.argmin(vdc, axis=1)
    worst_vals = vdc[np.arange(n), worst_idx]
    base = 1.0 - worst_vals
    base = np.clip(base, 0.0, 1.0)
    intent_on_worst = W[np.arange(n), worst_idx]
    return np.log1p((base * intent_on_worst).reshape(-1, 1))


def test_add_magnitude_matches_reference():
    clf = GMMClassifier(n_components=5, random_state=0, mag_weight=1.0)

    # Construct deterministic VDC and intent patterns
    X = np.array([
        [0.18, 0.18, 0.18],   # worst=0.18, W=[1,1,1] => base=0.82 * 1.0
        [0.12, 0.19, 0.19],   # worst=0.12 (V), W=[0.5,2.0,0.5] => base=0.88 * 0.5
        [0.15, 0.10, 0.19],   # worst=0.10 (D), W=[1.2,1.8,0.7] => base=0.90 * 1.8
        [0.16, 0.17, 0.14],   # worst=0.14 (C), W=[0.8,0.8,0.8] => base=0.86 * 0.8
    ], dtype=float)
    W = np.array([
        [1.0, 1.0, 1.0],
        [0.5, 2.0, 0.5],
        [1.2, 1.8, 0.7],
        [0.8, 0.8, 0.8],
    ], dtype=float)

    expected_mag = compute_expected_mag(X, W)

    X_with_mag = clf._add_magnitude(X, W)
    got_mag = X_with_mag[:, 3:4]

    assert np.allclose(got_mag, expected_mag, atol=1e-9), (
        f"Magnitude mismatch.\nExpected:\n{expected_mag}\nGot:\n{got_mag}"
    )


def test_cluster_severity_orders_by_magnitude_desc():
    # Build fake cluster means in RAW layout: [V,D,C,mag,cosV,cosD,cosC]
    # Only mag should drive primary ordering; we vary VDC/semantics arbitrarily
    means = np.array([
        [0.50, 0.50, 0.50, 0.10,  0.10,  0.10,  0.10],  # smallest mag
        [0.50, 0.48, 0.52, 0.70, -0.10, -0.05, -0.02],
        [0.52, 0.49, 0.50, 0.30,  0.05,  0.02,  0.01],
        [0.47, 0.51, 0.49, 0.90, -0.20, -0.10, -0.08],  # largest mag
        [0.51, 0.50, 0.49, 0.20,  0.00,  0.00,  0.00],
    ], dtype=float)

    clf = GMMClassifier(n_components=5, random_state=0)
    # Ensure no scaler is applied in mapping (means_raw == means)
    clf._mu_ = None
    clf._sigma_ = None

    mapping = clf._map_clusters_to_severity(means)

    # Identify indices by mag
    mag = means[:, 3]
    idx_max = int(np.argmax(mag))
    idx_min = int(np.argmin(mag))

    # Most severe (RED) should map to the largest magnitude cluster
    assert mapping[idx_max] == int(SeverityLevel.RED)
    # Least severe (BLUE) should map to the smallest magnitude cluster
    assert mapping[idx_min] == int(SeverityLevel.BLUE)
