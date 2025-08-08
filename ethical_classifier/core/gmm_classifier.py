"""
GMM Classifier for Ethical Content Analysis

Implements a Gaussian Mixture Model classifier for ethical content classification.
This version operates on 3D ethical vectors (Virtue, Deontological, Consequentialist)
and automatically incorporates magnitude as a fourth dimension for improved separation.

Author: Reconstructed from last-known-good framework
"""

from __future__ import annotations

import warnings
from enum import IntEnum
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.utils.validation import check_array

try:
    # Local aligner component
    from .spectrum_density_aligner import SpectrumDensityAligner
except Exception:  # pragma: no cover - allow import before aligner exists
    SpectrumDensityAligner = None  # type: ignore


class SeverityLevel(IntEnum):
    """Enumeration of severity levels for ethical classification.

    Note: WHITE and BLACK are theoretical perfect states that should never be
    assigned in practice, but are useful for ML boundaries.
    """

    WHITE = -1  # theoretical perfect good (never assigned)
    BLUE = 0  # strongly ethical
    GREEN = 1  # moderately ethical
    YELLOW = 2  # borderline
    ORANGE = 3  # moderately unethical
    RED = 4  # severely unethical
    BLACK = 5  # theoretical perfect evil (never assigned)

    @classmethod
    def get_name(cls, level: int) -> str:
        try:
            return cls(level).name.lower()
        except ValueError:
            if level < cls.WHITE:
                return "white"
            return "black"


class GMMClassifier:
    """
    Gaussian Mixture Model (GMM) based classifier for ethical evaluation.

    Operates on 3D VDC vectors and adds a 4th magnitude dimension to
    improve cluster separation. Maps clusters to practical severity
    levels (BLUE..RED).
    """

    def __init__(
        self,
        n_components: int = 5,
        use_spectrum_aligner: bool = True,
        random_state: Optional[int] = None,
        covariance_type: str = "full",
        tol: float = 1e-3,
        reg_covar: float = 1e-6,
        max_iter: int = 200,
        n_init: int = 10,
        init_params: str = "kmeans",
        weights_init: Optional[list] = None,
        means_init: Optional[np.ndarray] = None,
        precisions_init: Optional[np.ndarray] = None,
        warm_start: bool = False,
        verbose: int = 0,
        verbose_interval: int = 10,
        aligner_params: Optional[Dict[str, Any]] = None,
        mag_weight: float = 1.0,
        vdc_violation_threshold: float = 0.10,
        vdc_expected_max: float = 0.19,
        **gmm_kwargs: Any,
    ) -> None:
        if n_components < 2:
            raise ValueError("At least 2 components are required")

        self.n_components = n_components
        self.random_state = random_state
        self.covariance_type = covariance_type
        self.tol = tol
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.n_init = n_init
        self.init_params = init_params
        self.weights_init = weights_init
        self.means_init = means_init
        self.precisions_init = precisions_init
        self.warm_start = warm_start
        self.verbose = verbose
        self.verbose_interval = verbose_interval
        self.gmm_kwargs = gmm_kwargs

        # Weight for magnitude dimension in 4D features (pre-standardization)
        self.mag_weight: float = float(mag_weight)
        # Domain calibration (provided by service; used to compute magnitude correctly)
        self.vdc_violation_threshold: float = float(vdc_violation_threshold)
        self.vdc_expected_max: float = float(vdc_expected_max)

        self.model = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            tol=tol,
            reg_covar=reg_covar,
            max_iter=max_iter,
            n_init=n_init,
            init_params=init_params,
            weights_init=weights_init,
            means_init=means_init,
            precisions_init=precisions_init,
            random_state=random_state,
            warm_start=warm_start,
            verbose=verbose,
            verbose_interval=verbose_interval,
            **gmm_kwargs,
        )

        # Practical BLUE..RED range and state
        self.classes_ = np.arange(SeverityLevel.BLUE, SeverityLevel.RED + 1)
        self.is_fitted = False
        self.cluster_to_severity_: Optional[np.ndarray] = None
        self.ethical_component_: Optional[str] = "VDC+mag"  # marker for checks

        # Optional spectrum-density aligner
        self.use_spectrum_density = bool(use_spectrum_aligner)
        self.aligner: Optional[SpectrumDensityAligner] = None
        self.aligner_params: Dict[str, Any] = aligner_params or {}
        if self.use_spectrum_density and SpectrumDensityAligner is not None:
            try:
                self.aligner = SpectrumDensityAligner(**self.aligner_params)
            except Exception:
                # Fall back gracefully if dependencies not available
                self.aligner = None
                self.use_spectrum_density = False

        # Derived / cached
        self.practical_levels = int(SeverityLevel.RED - SeverityLevel.BLUE + 1)
        # Standardization parameters (computed on fit)
        self._mu_: Optional[np.ndarray] = None
        self._sigma_: Optional[np.ndarray] = None
        # Supervised class stats (for mapping/debug only)
        self._sup_classes_: Optional[np.ndarray] = None
        self._sup_means_: Optional[np.ndarray] = None

    # ----------------------------
    # Internals
    # ----------------------------
    def _add_magnitude(
        self, X: np.ndarray, intent_weights: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Return 4D features by appending a magnitude dimension.

        Magnitude encodes the domain rule that lower V/D/C values are more
        unethical. We use the worst (minimum) axis, scaled by the corresponding
        intent weight, and then apply a log1p transform for stability.
        If intent weights are given (shape (n, 3)), we pick the weight aligned
        with the worst axis per sample.
        """
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be 2D array")
        if X.shape[1] < 3:
            raise ValueError("X must have at least 3 columns for V, D, C")

        vdc = X[:, :3]
        # Preserve any additional semantic features beyond the first 3 (V,D,C)
        extras = X[:, 3:] if X.shape[1] > 3 else None
        n = len(vdc)
        # Determine worst axis (maximum V/D/C value) per sample — higher means more unethical in this domain
        worst_idx = np.argmax(vdc, axis=1)
        worst_vals = vdc[np.arange(n), worst_idx]
        # Threshold-relative severity: only count mass ABOVE violation threshold.
        # Domain: larger V/D/C magnitudes (toward ~0.19) are more unethical.
        thr = max(self.vdc_violation_threshold, 1e-6)
        vmax = max(self.vdc_expected_max, thr + 1e-6)
        # sev = (worst - thr) / (vmax - thr) in [0,1] when worst >= thr, else 0
        sev = (worst_vals - thr) / (vmax - thr)
        sev = np.clip(sev, 0.0, 1.0)

        # Align intent with the worst axis
        if intent_weights is not None and len(intent_weights) > 0:
            W = np.asarray(intent_weights)
            if W.ndim == 1:
                W = W.reshape(-1, 1)
            if W.shape[1] < 3:
                W = np.tile(W, (1, 3))
            n_eff = min(n, len(W))
            intent_on_worst = np.ones(n, dtype=float)
            intent_on_worst[:n_eff] = W[np.arange(n_eff), worst_idx[:n_eff]]
        else:
            intent_on_worst = np.ones(n, dtype=float)

        mag = np.log1p((sev * intent_on_worst).reshape(-1, 1))
        # Apply optional magnitude weight (default 1.0) prior to standardization
        mag = mag * float(self.mag_weight)
        if extras is not None and extras.size > 0:
            return np.hstack([vdc, mag, extras])
        return np.hstack([vdc, mag])

    def _map_clusters_to_severity(self, means: np.ndarray) -> np.ndarray:
        """Map GMM cluster means to practical severity levels (0..4).

        Semantics-aligned mapping: compute the ordering in RAW (de-standardized)
        VDC space and assign severities so that LOWER mean VDC => HIGHER severity.
        This preserves the domain rule that low V/D/C magnitudes indicate more
        unethical content on this scale.
        """
        # Inverse-standardize the means back to raw space if standardization stats exist
        try:
            if self._mu_ is not None and self._sigma_ is not None:
                means_raw = (np.asarray(means) * self._sigma_) + self._mu_
            else:
                means_raw = np.asarray(means)
        except Exception:
            means_raw = np.asarray(means)

        # Features layout: [V, D, C, mag, (optional) cosV, cosD, cosC, ...]
        n = means_raw.shape[0]
        mag = means_raw[:, 3] if means_raw.shape[1] >= 4 else np.zeros(n)
        # Primary ordering: DESC by magnitude (intent-weighted worst-axis proxy)
        order = np.argsort(-mag)

        # Optional tie-break: if adjacent clusters have very similar magnitude,
        # refine ordering using semantic similarity (lower mean cos => more severe)
        eps = getattr(self, "_mapping_mag_eps_", 1e-4)  # raw-scale epsilon for mag
        # If magnitude is essentially flat across clusters, fall back to VDC-based ordering
        if (np.max(mag) - np.min(mag)) <= eps:
            # Use the minimum V/D/C mean as proxy severity; higher values are more severe in this domain
            vdc_means = means_raw[:, :3]
            vdc_worst = np.min(vdc_means, axis=1)
            order = np.argsort(-vdc_worst)  # descending: highest worst-axis VDC first (most severe)
        use_semantic = means_raw.shape[1] >= 7  # cosV, cosD, cosC present
        if use_semantic and len(order) > 1:
            sem = means_raw[:, 4:7]
            sem_mean = np.mean(sem, axis=1)
            groups = []
            curr = [order[0]]
            for a, b in zip(order[:-1], order[1:]):
                if abs(mag[a] - mag[b]) <= eps:
                    curr.append(b)
                else:
                    groups.append(curr)
                    curr = [b]
            groups.append(curr)
            refined = []
            for g in groups:
                if len(g) == 1:
                    refined.extend(g)
                else:
                    # Lower semantic mean implies more harmful intent
                    g_sorted = sorted(g, key=lambda idx: sem_mean[idx])
                    refined.extend(g_sorted)
            order = np.array(refined, dtype=int)

        sev_order = [
            int(SeverityLevel.RED),
            int(SeverityLevel.ORANGE),
            int(SeverityLevel.YELLOW),
            int(SeverityLevel.GREEN),
            int(SeverityLevel.BLUE),
        ]
        mapping = np.zeros(len(means_raw), dtype=int)
        if len(order) == 5:
            for i, comp_idx in enumerate(order):
                mapping[comp_idx] = sev_order[i]
        else:
            # Interpolate severities across ranks if n_components != 5
            for rank, comp_idx in enumerate(order):
                # Map rank in [0..k-1] to index in [0..4] with reverse severity (low->high => RED->BLUE)
                if len(order) > 1:
                    frac = 1.0 - (rank / (len(order) - 1))
                else:
                    frac = 1.0
                idx = int(round(frac * (len(sev_order) - 1)))
                idx = max(0, min(idx, len(sev_order) - 1))
                mapping[comp_idx] = sev_order[idx]
        return mapping

    def _standardize_fit(self, X_4d: np.ndarray) -> np.ndarray:
        mu = X_4d.mean(axis=0)
        sigma = X_4d.std(axis=0)
        sigma[sigma < 1e-8] = 1e-8
        self._mu_ = mu
        self._sigma_ = sigma
        return (X_4d - mu) / sigma

    def _standardize_apply(self, X_4d: np.ndarray) -> np.ndarray:
        if self._mu_ is None or self._sigma_ is None:
            return X_4d
        return (X_4d - self._mu_) / self._sigma_

    def _check_is_fitted(self) -> None:
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model not fitted. Call 'fit' first.")

    # ----------------------------
    # Public API
    # ----------------------------
    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        intent_weights: Optional[np.ndarray] = None,
    ) -> "GMMClassifier":
        """Fit GMM to 3D VDC data, internally adding magnitude as 4th dim."""
        if self.is_fitted:
            warnings.warn("Model is already fitted. Re-fitting with new data.")
        X = check_array(X, ensure_min_samples=1, ensure_min_features=1)
        X_4d = self._add_magnitude(X, intent_weights)
        # Standardize feature space for stable EM
        X_4d_z = self._standardize_fit(X_4d)

        # Optionally prepare supervised initialization if labels are provided
        means_init = self.means_init
        weights_init = self.weights_init
        precisions_init = self.precisions_init
        if y is not None:
            try:
                y_arr = np.asarray(y).astype(int)
                classes = np.unique(y_arr)
                # Only attempt if component count matches number of classes
                if len(classes) == self.n_components:
                    means_4d = []
                    priors = []
                    diag_vars = []
                    for cls in sorted(classes):
                        idx = np.where(y_arr == cls)[0]
                        if len(idx) == 0:
                            mu4 = X_4d.mean(axis=0)
                            Zc = X_4d_z
                            pri = 1.0 / self.n_components
                        else:
                            mu4 = X_4d[idx].mean(axis=0)
                            Zc = X_4d_z[idx]
                            pri = len(idx) / len(y_arr)
                        means_4d.append(mu4)
                        priors.append(pri)
                        # diagonal variances on standardized data
                        v = Zc.var(axis=0)
                        v[v < 1e-6] = 1e-6
                        diag_vars.append(v)
                    means_init = np.vstack(means_4d)
                    weights_init = np.asarray(priors, dtype=float)
                    weights_init = weights_init / np.clip(weights_init.sum(), 1e-12, None)
                    diag_vars_arr = np.vstack(diag_vars)
                    # For diag covariance, precisions_init is (n_components, n_features)
                    if self.covariance_type == "diag":
                        precisions_init = 1.0 / diag_vars_arr
                    else:
                        # For 'full' or other types, let sklearn estimate precisions
                        precisions_init = None
            except Exception:
                means_init = self.means_init
                weights_init = self.weights_init
                precisions_init = self.precisions_init

        # If we will fit on standardized data, ensure means_init is standardized too
        if means_init is not None:
            try:
                means_init = self._standardize_apply(np.asarray(means_init))
            except Exception:
                pass

        # Rebuild model to ensure clean state
        init_params = self.init_params
        n_init = self.n_init
        if y is not None and means_init is not None:
            # Ensure sklearn uses provided means_init deterministically
            init_params = "random"
            n_init = 1
        self.model = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            tol=self.tol,
            reg_covar=self.reg_covar,
            max_iter=self.max_iter,
            n_init=n_init,
            init_params=init_params,
            weights_init=weights_init,
            means_init=means_init,
            precisions_init=precisions_init,
            random_state=self.random_state,
            warm_start=self.warm_start,
            verbose=self.verbose,
            verbose_interval=self.verbose_interval,
            **self.gmm_kwargs,
        )
        self.model.fit(X_4d_z)

        # Cluster-to-severity mapping
        if y is not None:
            # Collect supervised class means for diagnostics (not used for decision)
            y_arr = np.asarray(y).astype(int)
            self._sup_classes_ = np.array(sorted(np.unique(y_arr)))
            sup_means = []
            for cls in self._sup_classes_:
                idx = np.where(y_arr == cls)[0]
                if len(idx) == 0:
                    continue
                Zc = X_4d_z[idx]
                sup_means.append(Zc.mean(axis=0))
            if len(sup_means) == self.n_components:
                self._sup_means_ = np.vstack(sup_means)

            # Optimal 1-1 mapping using Hungarian assignment on component x class soft counts
            classes = list(self._sup_classes_) if self._sup_classes_ is not None else sorted(np.unique(y_arr))
            resp = self.model.predict_proba(X_4d_z)  # responsibilities shape (n_samples, n_components)
            counts = np.zeros((self.n_components, len(classes)), dtype=float)
            for col, cls in enumerate(classes):
                mask = (y_arr == int(cls))
                if np.any(mask):
                    # Sum responsibilities over samples of this class
                    counts[:, col] = resp[mask].sum(axis=0)
            mapping = np.zeros(self.n_components, dtype=int)
            try:
                from scipy.optimize import linear_sum_assignment  # type: ignore
                # Maximize counts => minimize negative counts
                row_ind, col_ind = linear_sum_assignment(-counts)
                for r, c in zip(row_ind, col_ind):
                    mapping[r] = int(classes[c])
                # Any components not assigned (e.g., empty) fall back to unsupervised order
                default_map = self._map_clusters_to_severity(self.model.means_)
                for comp in range(self.n_components):
                    if mapping[comp] not in classes:
                        mapping[comp] = int(default_map[comp])
                # Validate mapping on training data; fallback if suspicious
                self.cluster_to_severity_ = mapping
                try:
                    train_labels = self.model.predict(X_4d_z)
                    preds = np.array([self.cluster_to_severity_[lbl] for lbl in train_labels])
                    # Normalize y into 0..4 if needed
                    y_norm = y_arr.astype(int)
                    tr_acc = (preds == y_norm).mean()
                    if tr_acc < 0.80:
                        # Fallback to ordering by mean severity score
                        self.cluster_to_severity_ = self._map_clusters_to_severity(self.model.means_)
                except Exception:
                    pass
            except Exception:
                # Fallback to majority vote per component
                default_map = self._map_clusters_to_severity(self.model.means_)
                for comp in range(self.n_components):
                    col = int(np.argmax(counts[comp])) if counts.shape[1] > 0 else 0
                    if counts[comp, col] <= 0:
                        mapping[comp] = int(default_map[comp])
                    else:
                        mapping[comp] = int(classes[col])
                self.cluster_to_severity_ = mapping
        else:
            # Fallback: unsupervised ordering
            self.cluster_to_severity_ = self._map_clusters_to_severity(self.model.means_)

        # Optional spectrum-density aligner
        if self.use_spectrum_density and self.aligner is not None:
            self.aligner.fit(X_4d_z, gmm_centers=self.model.means_)

        self.is_fitted = True
        return self

    def predict_proba(
        self,
        X: np.ndarray,
        intent_weights: Optional[np.ndarray] = None,
        use_alignment: Optional[bool] = None,
    ) -> np.ndarray:
        """Return probabilities over practical severity levels (0..4)."""
        self._check_is_fitted()
        X_4d = self._add_magnitude(X, intent_weights)
        X_4d = self._standardize_apply(X_4d)
        # Pure GMM probability aggregation path
        # Default: aggregate GMM component probabilities
        comp_probs = self.model.predict_proba(X_4d)

        # Default alignment behavior
        use_alignment = self.use_spectrum_density if use_alignment is None else use_alignment
        if use_alignment and self.aligner is not None:
            gmm_labels = self.model.predict(X_4d)
            levels = self._map_to_severity_levels(gmm_labels, X_4d, use_alignment=True)
            out = np.zeros((len(X), self.practical_levels))
            out[np.arange(len(X)), levels] = 1.0
            return out

        # Aggregate component probabilities by mapped severity
        out = np.zeros((len(X), self.practical_levels))
        assert self.cluster_to_severity_ is not None
        for comp_idx, sev in enumerate(self.cluster_to_severity_):
            out[:, sev] += comp_probs[:, comp_idx]
        # Enforce zero-vector -> BLUE policy
        try:
            vdc = np.asarray(X)[:, :3]
            eps = 1e-12
            zero_mask = np.all(np.abs(vdc) < eps, axis=1)
            if np.any(zero_mask):
                out[zero_mask, :] = 0.0
                out[zero_mask, int(SeverityLevel.BLUE)] = 1.0
        except Exception:
            pass
        return out

    def _map_to_severity_levels(
        self, cluster_indices: np.ndarray, X_4d: np.ndarray, use_alignment: bool = False
    ) -> np.ndarray:
        if self.cluster_to_severity_ is None or len(self.cluster_to_severity_) != self.n_components:
            self.cluster_to_severity_ = np.arange(self.n_components) % self.practical_levels
        if use_alignment and self.aligner is not None:
            aligned_labels, _ = self.aligner.predict(X_4d, cluster_indices)
            # Map aligned clusters to nearest GMM centers, then to severity
            from sklearn.metrics import pairwise_distances_argmin_min

            unique = np.unique(aligned_labels)
            centers = np.array([np.median(X_4d[aligned_labels == i], axis=0) for i in unique])
            nearest_gmm = pairwise_distances_argmin_min(centers, self.model.means_)[0]
            aligned_to_sev = {label: self.cluster_to_severity_[g] for label, g in zip(unique, nearest_gmm)}
            return np.array([aligned_to_sev[label] for label in aligned_labels])
        else:
            return np.array([self.cluster_to_severity_[idx] for idx in cluster_indices])

    def predict(
        self,
        X: np.ndarray,
        intent_weights: Optional[np.ndarray] = None,
        return_severity_name: bool = False,
        use_alignment: Optional[bool] = None,
        return_confidence: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Predict severity levels for samples in X.

        Returns integers 0..4 (BLUE..RED) or names if return_severity_name.
        If return_confidence, also returns normalized confidence scores.
        """
        self._check_is_fitted()
        X_4d = self._add_magnitude(X, intent_weights)
        X_4d = self._standardize_apply(X_4d)
        # Pure GMM decision path

        gmm_labels = self.model.predict(X_4d)
        log_probs = self.model.score_samples(X_4d)
        min_lp, max_lp = float(np.min(log_probs)), float(np.max(log_probs))
        conf = (log_probs - min_lp) / (max_lp - min_lp) if max_lp > min_lp else np.ones_like(log_probs)

        use_alignment = self.use_spectrum_density if use_alignment is None else use_alignment
        levels = self._map_to_severity_levels(gmm_labels, X_4d, use_alignment)
        # Enforce zero-vector -> BLUE policy
        try:
            vdc = np.asarray(X)[:, :3]
            eps = 1e-12
            zero_mask = np.all(np.abs(vdc) < eps, axis=1)
            if np.any(zero_mask):
                levels = np.asarray(levels).copy()
                levels[zero_mask] = int(SeverityLevel.BLUE)
        except Exception:
            pass

        if return_severity_name:
            names = np.array([SeverityLevel.get_name(int(l)) for l in levels])
            return (names, conf) if return_confidence else names
        return (levels, conf) if return_confidence else levels

    # Convenience accessors
    def get_cluster_means(self) -> np.ndarray:
        self._check_is_fitted()
        return self.model.means_

    def get_params(self) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "n_components": self.n_components,
            "covariance_type": self.covariance_type,
            "reg_covar": self.reg_covar,
            "max_iter": self.max_iter,
            "random_state": self.random_state,
            **self.gmm_kwargs,
        }
        return params
