"""
Monotonic Ethics Service for the Ethical AI Testbed.

This service implements a monotonic, signed-strata classifier for ethical evaluation
based on the core axiom of maximizing human autonomy within objective empirical truth.

The service integrates:
1. Signed VDC transformation from violation-pointing to ethical-pointing space
2. Monotonic classification with intent amplification
3. Future-ready clustering experimentation capabilities
"""

import logging
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Union
from pathlib import Path
import os
import asyncio

from ethical_classifier.core.signed_vdc_transformer import SignedVDCTransformer
from ethical_classifier.core.monotonic_classifier import MonotonicClassifier, MonotonicClusteringClassifier

from application.services.vector_generation_service import VectorGenerationService
from application.services.intent_analysis_service import IntentAnalysisService

logger = logging.getLogger(__name__)

class MonotonicEthicsService:
    """
    Service for monotonic ethical evaluation with signed VDC vectors.
    
    This service integrates the monotonic classifier with the existing
    vector generation and intent analysis services to provide ethical
    evaluation capabilities with strict monotonicity guarantees.
    
    The service:
    1. Obtains raw VDC vectors from the existing VectorGenerationService
    2. Transforms them to signed [-1, +1] space using SignedVDCTransformer
    3. Applies monotonic classification with optional intent amplification
    4. Supports future clustering experiments (disabled by default)
    
    Attributes:
        vector_service: Service for generating VDC vectors
        intent_service: Service for analyzing intent
        transformer: Transforms VDC vectors to signed space
        classifier: Monotonic classifier for ethical evaluation
        model_path: Path to saved classifier model
        enable_clustering: Whether to use clustering (future experimentation)
        clustering_algorithm: Clustering algorithm to use
    """
    
    def __init__(self, 
                 vector_service: Optional[VectorGenerationService] = None,
                 intent_service: Optional[IntentAnalysisService] = None,
                 model_path: Optional[str] = None,
                 enable_clustering: bool = False,
                 clustering_algorithm: str = 'kmeans_ordinal'):
        """
        Initialize the monotonic ethics service.
        
        Args:
            vector_service: Service for generating VDC vectors
            intent_service: Service for analyzing intent
            model_path: Path to saved classifier model
            enable_clustering: Whether to use clustering
            clustering_algorithm: Clustering algorithm to use
        """
        # Initialize vector and intent services
        self.vector_service = vector_service or VectorGenerationService()
        self.intent_service = intent_service or IntentAnalysisService()
        
        # Initialize transformer and classifier
        self.transformer = SignedVDCTransformer(
            violation_threshold=0.10,  # VDC threshold for ethical/unethical boundary
            max_observed=0.19          # Maximum observed VDC value
        )
        
        # Default model path if not specified
        if model_path is None:
            model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                                     "models", "monotonic_classifier.joblib")
        
        self.model_path = model_path
        self.enable_clustering = enable_clustering
        self.clustering_algorithm = clustering_algorithm
        
        # Initialize classifier
        if enable_clustering:
            self.classifier = MonotonicClusteringClassifier()
            logger.info("Initialized MonotonicEthicsService with clustering capabilities")
        else:
            self.classifier = MonotonicClassifier()
            logger.info("Initialized MonotonicEthicsService with monotonic classifier")
            
        # Try to load saved model
        self._load_model_if_exists()
    
    def _load_model_if_exists(self) -> bool:
        """
        Load classifier model if it exists.
        
        Returns:
            Whether a model was loaded
        """
        try:
            if os.path.exists(self.model_path):
                if self.enable_clustering:
                    self.classifier = MonotonicClusteringClassifier.load(self.model_path)
                else:
                    self.classifier = MonotonicClassifier.load(self.model_path)
                logger.info(f"Loaded monotonic classifier from {self.model_path}")
                return True
        except Exception as e:
            logger.warning(f"Failed to load monotonic classifier from {self.model_path}: {str(e)}")
        
        return False
    
    async def _get_raw_vdc_vectors(self, texts: List[str]) -> np.ndarray:
        """
        Get raw VDC vectors from the vector service.
        
        Args:
            texts: List of texts to get vectors for
            
        Returns:
            Array of shape (n_samples, 3) with raw VDC vectors
        """
        # Get vectors from vector generation service
        text_to_vector = {}
        
        for text in texts:
            # Use existing asynchronous API if available
            vector = await self.vector_service.get_vdc_vector(text)
            text_to_vector[text] = vector
            
        # Convert to numpy array
        raw_vdc = np.array([text_to_vector[text] for text in texts])
        
        logger.debug(f"Retrieved {len(texts)} raw VDC vectors, "
                    f"range: [{raw_vdc.min():.3f}, {raw_vdc.max():.3f}]")
        return raw_vdc
    
    async def _get_intent_weights(self, texts: List[str]) -> np.ndarray:
        """
        Get intent weights from the intent service.
        
        Args:
            texts: List of texts to get intent weights for
            
        Returns:
            Array of shape (n_samples,) with intent weights
        """
        intent_weights = []
        
        for text in texts:
            try:
                # Get intent from intent analysis service
                intent_result = await self.intent_service.analyze_intent(text)
                
                # Extract harmful intent score (0-1)
                harm_score = intent_result.get('harm_score', 0.0)
                intent_weights.append(harm_score)
            except Exception as e:
                logger.warning(f"Failed to get intent for '{text[:30]}...': {str(e)}")
                intent_weights.append(0.0)  # Default: no harmful intent
        
        return np.array(intent_weights)
    
    async def predict(self, texts: List[str], parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Predict ethical evaluation for texts.
        
        Args:
            texts: List of texts to evaluate
            parameters: Optional parameters for customizing evaluation
                - use_intent: Whether to use intent amplification (default: True)
                - use_clustering: Whether to use clustering refinement (default: False)
                - clustering_algorithm: Clustering algorithm to use (default: 'kmeans_ordinal')
                
        Returns:
            Dictionary with prediction results:
            - texts: List of input texts
            - strata: List of severity levels (0=BLUE, 1=GREEN, ..., 4=RED)
            - strata_names: List of severity level names
            - strata_colors: List of severity level colors
            - confidence: List of confidence scores
            - signed_vdc: List of signed VDC vectors
        """
        if not texts:
            return {"error": "No texts provided"}
            
        # Default parameters
        params = {
            "use_intent": True,
            "use_clustering": self.enable_clustering,
            "clustering_algorithm": self.clustering_algorithm
        }
        
        # Update with user parameters
        if parameters:
            params.update(parameters)
        
        # Get raw VDC vectors
        raw_vdc = await self._get_raw_vdc_vectors(texts)
        
        # Transform to signed space
        signed_vdc = self.transformer.transform_to_signed(raw_vdc)
        
        # Get intent weights if requested
        intent_weights = None
        if params["use_intent"]:
            intent_weights = await self._get_intent_weights(texts)
        
        # Make predictions
        if params["use_clustering"] and isinstance(self.classifier, MonotonicClusteringClassifier):
            strata, confidence, cluster_info = self.classifier.predict_with_clustering(
                signed_vdc, intent_weights, params["clustering_algorithm"]
            )
            result = {
                "texts": texts,
                "strata": strata.tolist(),
                "strata_names": [self.classifier.strata_names[s] for s in strata],
                "strata_colors": [self.classifier.strata_colors[s] for s in strata],
                "confidence": confidence.tolist(),
                "signed_vdc": signed_vdc.tolist(),
                "cluster_info": cluster_info
            }
        else:
            strata, confidence = self.classifier.predict_strata(signed_vdc, intent_weights)
            result = {
                "texts": texts,
                "strata": strata.tolist(),
                "strata_names": [self.classifier.strata_names[s] for s in strata],
                "strata_colors": [self.classifier.strata_colors[s] for s in strata],
                "confidence": confidence.tolist(),
                "signed_vdc": signed_vdc.tolist()
            }
        
        # Log distribution summary
        counts = np.bincount(strata, minlength=self.classifier.n_strata)
        distribution = {
            self.classifier.strata_names[i]: int(counts[i])
            for i in range(self.classifier.n_strata)
        }
        logger.info(f"Prediction distribution: {distribution}")
        
        return result
    
    async def train(self, texts: List[str], labels: List[int],
                  parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Train the monotonic classifier with labeled data.
        
        Args:
            texts: List of texts for training
            labels: List of severity levels (0=BLUE, 1=GREEN, ..., 4=RED)
            parameters: Optional parameters for customizing training
                - use_intent: Whether to use intent in training (default: True)
                
        Returns:
            Dictionary with training results:
            - success: Whether training was successful
            - thresholds: Learned severity thresholds
            - accuracy: Training accuracy
        """
        if not texts or not labels or len(texts) != len(labels):
            return {"error": "Invalid training data"}
            
        # Default parameters
        params = {"use_intent": True}
        
        # Update with user parameters
        if parameters:
            params.update(parameters)
        
        # Get raw VDC vectors
        raw_vdc = await self._get_raw_vdc_vectors(texts)
        
        # Transform to signed space
        signed_vdc = self.transformer.transform_to_signed(raw_vdc)
        
        # Get intent weights if requested
        intent_weights = None
        if params["use_intent"]:
            intent_weights = await self._get_intent_weights(texts)
        
        # Convert labels to numpy array
        labels_array = np.array(labels)
        
        # Learn thresholds
        thresholds = self.classifier.learn_thresholds(signed_vdc, labels_array, intent_weights)
        
        # Evaluate accuracy
        strata, _ = self.classifier.predict_strata(signed_vdc, intent_weights)
        accuracy = np.mean(strata == labels_array)
        
        # Save model
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        self.classifier.save(self.model_path)
        
        return {
            "success": True,
            "thresholds": thresholds.tolist(),
            "accuracy": float(accuracy)
        }
    
    async def save(self) -> Dict[str, Any]:
        """
        Save the classifier model.
        
        Returns:
            Dictionary with save results
        """
        try:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            self.classifier.save(self.model_path)
            return {"success": True, "path": self.model_path}
        except Exception as e:
            logger.error(f"Failed to save monotonic classifier: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def load(self) -> Dict[str, Any]:
        """
        Load the classifier model.
        
        Returns:
            Dictionary with load results
        """
        try:
            if not os.path.exists(self.model_path):
                return {"success": False, "error": "Model file not found"}
                
            if self.enable_clustering:
                self.classifier = MonotonicClusteringClassifier.load(self.model_path)
            else:
                self.classifier = MonotonicClassifier.load(self.model_path)
                
            return {"success": True, "path": self.model_path}
        except Exception as e:
            logger.error(f"Failed to load monotonic classifier: {str(e)}")
            return {"success": False, "error": str(e)}
