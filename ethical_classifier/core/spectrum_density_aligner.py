"""
Spectrum and Density Aligner for Ethical Classification

Combines spectral clustering and density-based clustering to enhance GMM classification
by handling outliers and non-linear patterns in ethical vector space.

Author: Reconstructed from last-known-good framework
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.cluster import DBSCAN, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.neighbors import NearestNeighbors

try:  # optional dependency
    import hdbscan  # type: ignore
    HDBSCAN_AVAILABLE = True
except Exception:  # pragma: no cover
    hdbscan = None  # type: ignore
    HDBSCAN_AVAILABLE = False


class SpectrumDensityAligner:
    """
    Combines spectral and density-based clustering to enhance GMM classification.

    This component:
    1. Uses density-based clustering to identify and handle outliers
    2. Applies spectral clustering to capture non-linear patterns
    3. Integrates both with GMM for final classification
    """

    def __init__(
        self,
        # Density analyzer params
        density_algorithm: str = "hdbscan",
        eps: float = 0.5,
        min_samples: int = 5,
        min_cluster_size: int = 5,
        # Spectral analyzer params
        n_spectral_clusters: Optional[int] = None,
        affinity: str = "rbf",
        gamma: float = 1.0,
        n_neighbors: int = 10,
        # General params
        random_state: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        self.density_algorithm = density_algorithm
        self.eps = eps
        self.min_samples = min_samples
        self.min_cluster_size = min_cluster_size

        self.n_spectral_clusters = n_spectral_clusters
        self.affinity = affinity
        self.gamma = gamma
        self.n_neighbors = n_neighbors
        self.random_state = random_state

        # Runtime state
        self.density_analyzer = None
        self.spectral_analyzer: Optional[SpectralClustering] = None
        self.spectral_labels_: Optional[np.ndarray] = None
        self.gmm_centers_: Optional[np.ndarray] = None
        self.is_fitted: bool = False

    def _init_density_analyzer(self, X: np.ndarray) -> None:
        if self.density_algorithm == "hdbscan" and HDBSCAN_AVAILABLE:
            self.density_analyzer = hdbscan.HDBSCAN(
                min_cluster_size=self.min_cluster_size,
                min_samples=self.min_samples,
                core_dist_n_jobs=-1,
            )
            self.density_analyzer.fit(X)
        else:
            # Fallback to DBSCAN if HDBSCAN not available
            self.density_analyzer = DBSCAN(eps=self.eps, min_samples=self.min_samples)
            self.density_analyzer.fit(X)

    def _init_spectral_analyzer(self, X: np.ndarray) -> None:
        self.spectral_analyzer = SpectralClustering(
            n_clusters=self.n_spectral_clusters,
            affinity=self.affinity,
            gamma=self.gamma,
            n_neighbors=self.n_neighbors,
            random_state=self.random_state,
        )
        self.spectral_labels_ = self.spectral_analyzer.fit_predict(X)

    def fit(self, X: np.ndarray, gmm_centers: np.ndarray) -> "SpectrumDensityAligner":
        self.gmm_centers_ = gmm_centers
        if self.n_spectral_clusters is None:
            self.n_spectral_clusters = len(gmm_centers)
        self._init_density_analyzer(X)
        self._init_spectral_analyzer(X)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray, gmm_labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        self._check_is_fitted()

        # Density-based labels (-1 indicates noise for HDBSCAN/DBSCAN)
        if hasattr(self.density_analyzer, "fit_predict"):
            density_labels = self.density_analyzer.fit_predict(X)
        else:
            density_labels = self.density_analyzer.labels_  # type: ignore[attr-defined]
        is_noise = density_labels == -1

        # Spectral labels for new data
        if hasattr(self.spectral_analyzer, "fit_predict"):
            spectral_labels = self.spectral_analyzer.fit_predict(X)  # type: ignore[union-attr]
        else:
            # Fallback for unforeseen cases: nearest-neighbor transfer
            nbrs = NearestNeighbors(n_neighbors=1).fit(X)
            _, indices = nbrs.kneighbors(X)
            spectral_labels = self.spectral_labels_[indices.flatten()]  # type: ignore[index]

        # Initialize refined labels with GMM labels
        refined = gmm_labels.copy()

        if np.any(is_noise):
            noise_idx = np.where(is_noise)[0]
            noise_pts = X[noise_idx]
            nearest = pairwise_distances_argmin_min(noise_pts, self.gmm_centers_)[0]
            refined[noise_idx] = nearest

            # Optionally leverage spectral structure to smooth assignments
            for spec_cluster in np.unique(spectral_labels):
                mask = spectral_labels == spec_cluster
                if np.sum(mask) > 0:
                    common = np.argmax(np.bincount(refined[mask]))
                    refined[mask & is_noise] = common

        return refined, is_noise

    def _check_is_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("SpectrumDensityAligner is not fitted. Call 'fit' first.")

    # Convenience utilities
    def get_spectral_embedding(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        if getattr(self.spectral_analyzer, "embedding_", None) is not None:  # type: ignore[union-attr]
            return self.spectral_analyzer.embedding_  # type: ignore[union-attr]
        # Fallback to PCA
        n_components = min(5, X.shape[1])
        return PCA(n_components=n_components).fit_transform(X)

    def get_density_scores(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        if self.density_analyzer is None:
            nbrs = NearestNeighbors(n_neighbors=2).fit(X)
            distances, _ = nbrs.kneighbors(X)
            return distances[:, 1]
        if hasattr(self.density_analyzer, "outlier_scores_"):
            return self.density_analyzer.outlier_scores_
        if hasattr(self.density_analyzer, "core_sample_distances_"):
            return self.density_analyzer.core_sample_distances_
        if hasattr(self.density_analyzer, "labels_"):
            labels = self.density_analyzer.labels_
            if -1 in labels:
                non_noise = X[labels != -1]
                if len(non_noise) > 0:
                    nbrs = NearestNeighbors(n_neighbors=1).fit(non_noise)
                    distances, _ = nbrs.kneighbors(X)
                    return distances.flatten()
        # Final fallback
        nbrs = NearestNeighbors(n_neighbors=2).fit(X)
        distances, _ = nbrs.kneighbors(X)
        return distances[:, 1]
