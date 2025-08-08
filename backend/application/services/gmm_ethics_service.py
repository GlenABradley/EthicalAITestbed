"""
GMM Ethics Service

Provides training and inference for the GMM-based ethical classifier using
pipeline-grade intent vectors and optional spectrum-density alignment.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import os

import numpy as np

from ethical_classifier.core.gmm_classifier import GMMClassifier, SeverityLevel
from backend.core.evaluation_engine import OptimizedEvaluationEngine
from backend.application.services.intent_hierarchy_service import IntentHierarchyService
from backend.application.services.intent_hierarchy_service import LORA_AVAILABLE
from sentence_transformers import SentenceTransformer
from backend.application.services.vector_generation_service import VectorGenerationService
import joblib

logger = logging.getLogger(__name__)

class GMMEthicsService:
    def __init__(
        self,
        n_components: int = 5,
        random_state: Optional[int] = 123,
        use_spectrum_aligner: bool = True,
        mag_weight: float = 1.0,
        beta_intent: float = 1.2,
        covariance_type: str = "diag",
    ) -> None:
        # Core components
        self.engine = OptimizedEvaluationEngine()
        base_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.intent_hierarchy = IntentHierarchyService(base_model)
        self.base_model = base_model  # used for sentence embeddings

        # Compile fixed, code-level perspective vectors from the embedding statement
        vg = VectorGenerationService()  # uses its own embedding service under the hood
        p_v, p_d, p_c = vg.get_all_vectors()
        # Normalize for cosine similarity
        def _norm(v: np.ndarray) -> np.ndarray:
            n = np.linalg.norm(v)
            return v / n if n > 0 else v
        self.p_v = _norm(p_v.astype(np.float32))
        self.p_d = _norm(p_d.astype(np.float32))
        self.p_c = _norm(p_c.astype(np.float32))
        # Weight for semantic features appended to VDC+mag (disabled by default for stability)
        self.embedding_alpha: float = 0.0

        # Vector scale calibration (domain): vectors typically in [0.0, ~0.19], violations begin ≈0.10
        self.vdc_violation_threshold: float = 0.10
        self.vdc_expected_max: float = 0.19

        # Classifier (aligner enabled by default)
        self.clf = GMMClassifier(
            n_components=n_components,
            random_state=random_state,
            use_spectrum_aligner=use_spectrum_aligner,
            mag_weight=mag_weight,
            covariance_type=covariance_type,
            max_iter=300,
            n_init=5,
            verbose=0,
            vdc_violation_threshold=self.vdc_violation_threshold,
            vdc_expected_max=self.vdc_expected_max,
        )

        self.beta_intent = float(beta_intent)
        # Only apply harmful-intent amplification when confidence is sufficiently high
        self.harm_activation_threshold: float = 0.6
        # For UI stability, keep prediction alignment off (it can vary across batch sizes)
        self.predict_use_alignment: bool = False

    # ------------------------
    # Persistence
    # ------------------------
    def save_model(self, filepath: str) -> str:
        """Persist the trained classifier to disk. Returns absolute path saved.

        Only the underlying classifier is persisted (not the engine or intent services).
        """
        # Do not allow persisting an unfitted model; this causes confusing runtime errors later
        if not getattr(self.clf, "is_fitted", False):
            raise RuntimeError("Cannot save GMM model: classifier is not fitted. Train or load a fitted model first.")
        path = Path(filepath)
        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.clf, path)
        return str(path.resolve())

    def load_model(self, filepath: str) -> None:
        """Load a classifier from disk into this service instance."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        self.clf = joblib.load(path)
        # Backward compatibility: older persisted models may not have domain calibration attrs
        if not hasattr(self.clf, "vdc_violation_threshold"):
            setattr(self.clf, "vdc_violation_threshold", float(self.vdc_violation_threshold))
        if not hasattr(self.clf, "vdc_expected_max"):
            setattr(self.clf, "vdc_expected_max", float(self.vdc_expected_max))
        # Recompute mapping using current severity mapping rule in case it changed across versions
        try:
            if getattr(self.clf, "model", None) is not None and getattr(self.clf.model, "means_", None) is not None:
                self.clf.cluster_to_severity_ = self.clf._map_clusters_to_severity(self.clf.model.means_)
                # Ensure fitted flag is true for loaded, trained models
                self.clf.is_fitted = True
        except Exception:
            pass

    # ------------------------
    # Feature construction
    # ------------------------
    async def _text_to_vdc(self, text: str, parameters: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """Extract raw VDC scores (virtue, deontological, consequentialist) from the engine."""
        evaluation = await self.engine.evaluate_text_async(text, parameters)
        if not evaluation.spans:
            # On this scale, neutral/none should be near zero, not 0.5
            return np.array([0.0, 0.0, 0.0], dtype=float)

        def framework_specific_score(scores: List[float]) -> float:
            """Weighted mean that emphasizes larger (more unethical) values on 0..~0.19 scale.
            Defaults to 0.0 (ethical) if empty.
            """
            if not scores:
                return 0.0
            vmax = max(self.vdc_expected_max, 1e-6)
            total_weight = 0.0
            weighted_sum = 0.0
            for s in scores:
                # Emphasize larger values; weight grows linearly with s relative to expected max
                w = 1.0 + max(s, 0.0) / vmax
                weighted_sum += s * w
                total_weight += w
            return (weighted_sum / total_weight) if total_weight > 0 else 0.0

        v_scores = [span.virtue_score for span in evaluation.spans]
        d_scores = [span.deontological_score for span in evaluation.spans]
        c_scores = [span.consequentialist_score for span in evaluation.spans]

        v = framework_specific_score(v_scores)
        d = framework_specific_score(d_scores)
        c = framework_specific_score(c_scores)
        return np.array([v, d, c], dtype=float)

    def _intent_weights_from_text(self, vdc: np.ndarray, text: str) -> np.ndarray:
        intent_scores = self.intent_hierarchy.classify_intent(text)
        harm_intensity = max(intent_scores.values()) if intent_scores else 0.0
        # If LoRA adapters are unavailable, fallback classifiers are untrained → random.
        # To avoid instability, neutralize intent amplification.
        if not LORA_AVAILABLE:
            harm_intensity = 0.0
        # Suppress weak/noisy intent; only amplify when clearly harmful
        if harm_intensity < self.harm_activation_threshold:
            harm_intensity = 0.0
        # Per-dimension weights using harm intensity and deviation from violation threshold (≈0.10)
        # We want: low VDC (more unethical) + high harm intent -> INCREASE magnitude
        # and high VDC (more ethical) + high harm intent -> DECREASE magnitude slightly.
        thr = self.vdc_violation_threshold
        vmax = max(self.vdc_expected_max, 1e-6)
        # Use inverted deviation so that values BELOW threshold yield POSITIVE deviation.
        # dev in approx [-1, +1]
        dev = (thr - vdc) / vmax
        W = 1.0 + self.beta_intent * harm_intensity * dev
        return np.clip(W, 0.6, 2.0)

    async def _features_for_texts(self, texts: List[str], parameters: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, np.ndarray]:
        vdc_list: List[np.ndarray] = []
        W_list: List[np.ndarray] = []
        # Parallelize VDC extraction
        vdc_vals = await asyncio.gather(*(self._text_to_vdc(t, parameters) for t in texts))
        for vdc, text in zip(vdc_vals, texts):
            vdc_list.append(vdc)
            W_list.append(self._intent_weights_from_text(vdc, text))
        X = np.vstack(vdc_list).astype(float)
        W = np.vstack(W_list).astype(float)
        # Append perspective-aware semantic features from fixed statement
        if getattr(self, "embedding_alpha", 0.0) and abs(self.embedding_alpha) > 1e-12:
            try:
                # Compute sentence embeddings in a batch
                text_emb = self.base_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
                # Normalize for cosine similarity
                norms = np.linalg.norm(text_emb, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                text_unit = text_emb / norms
                # Cosine sims with each perspective vector
                cos_v = text_unit @ self.p_v.reshape(-1, 1)
                cos_d = text_unit @ self.p_d.reshape(-1, 1)
                cos_c = text_unit @ self.p_c.reshape(-1, 1)
                sims = np.hstack([cos_v, cos_d, cos_c]).astype(np.float32)
                if np.any(np.isnan(sims)):
                    sims = np.nan_to_num(sims, nan=0.0)
                # Scale and append
                X = np.hstack([X, self.embedding_alpha * sims])
            except Exception:
                # If embeddings fail for any reason, proceed with VDC-only features
                pass
        return X, W

    # ------------------------
    # Public API
    # ------------------------
    async def train(self, texts: List[str], labels: List[int], parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        X, W = await self._features_for_texts(texts, parameters)
        y = np.array(labels, dtype=int)
        self.clf.fit(X, y=y, intent_weights=W)
        # Report simple resubstitution accuracy and mapping for transparency
        y_hat = self.clf.predict(X, intent_weights=W)
        acc = float((y_hat == y).mean()) if len(y) else 0.0

        # Diagnostic: log cluster means in RAW space and mapping
        try:
            means_z = self.clf.get_cluster_means()
            mu = getattr(self.clf, "_mu_", None)
            sigma = getattr(self.clf, "_sigma_", None)
            means_raw = means_z * sigma + mu if (mu is not None and sigma is not None) else means_z
            mapping = getattr(self.clf, "cluster_to_severity_", None)
            logger.info("GMM cluster means (raw): V,D,C,mag,cosV,cosD,cosC ...\n%s", np.round(means_raw, 6))
            logger.info("GMM cluster->severity mapping: %s", mapping)
        except Exception as e:
            logger.warning("Failed to log cluster diagnostics: %s", e)
        return {
            "status": "trained",
            "train_accuracy": acc,
            "n_samples": int(len(y)),
        }

    async def predict(self, texts: List[str], parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        X, W = await self._features_for_texts(texts, parameters)
        # For stability across calls with varying batch sizes, disable alignment in prediction.
        # Training can still use the aligner internally.
        use_align = bool(self.predict_use_alignment)
        y_hat = self.clf.predict(X, intent_weights=W, use_alignment=use_align)
        proba = self.clf.predict_proba(X, intent_weights=W, use_alignment=use_align)
        # Optional per-text debug to verify intent multiplier direction
        if parameters and parameters.get("debug_intent"):
            try:
                # Compute 4D features (with magnitude) to log exact mag used by the classifier
                X4 = self.clf._add_magnitude(X, W)
                for i, text in enumerate(texts):
                    vdc = X[i, :3]
                    mag = X4[i, 3] if X4.shape[1] >= 4 else float('nan')
                    # worst-axis index for transparency
                    worst_idx = int(np.argmin(vdc))
                    # Recompute intent pieces for log clarity
                    raw_scores = self.intent_hierarchy.classify_intent(text)
                    hi_raw = max(raw_scores.values()) if raw_scores else 0.0
                    hi = hi_raw if hi_raw >= self.harm_activation_threshold else 0.0
                    thr = self.vdc_violation_threshold
                    vmax = max(self.vdc_expected_max, 1e-6)
                    dev = (thr - vdc) / vmax
                    logger.debug(
                        "[INTENT DEBUG] text=%r vdc=%s worst_axis=%d mag=%.6f harm_intensity_raw=%.3f harm_intensity_used=%.3f dev=%s W=%s pred=%s conf=%.3f",
                        text,
                        np.round(vdc, 6),
                        worst_idx,
                        float(mag),
                        hi_raw,
                        hi,
                        np.round(dev, 6),
                        np.round(W[i], 6),
                        SeverityLevel.get_name(int(y_hat[i])),
                        float(np.max(proba[i]) if proba is not None else 1.0),
                    )
            except Exception as e:
                logger.warning("Failed intent debug logging: %s", e)
        names = [SeverityLevel.get_name(int(v)) for v in y_hat]
        conf = np.max(proba, axis=1) if proba is not None else np.ones(len(y_hat))
        return {
            "predictions": y_hat.tolist(),
            "severity_names": names,
            "confidence": conf.tolist(),
        }
