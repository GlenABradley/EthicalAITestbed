"""
Configuration Service for the Ethical AI Testbed.

This service provides centralized configuration management for the ethical evaluation system,
including parameter validation, persistence, and dynamic updates.
"""

import logging
from typing import Dict, Any, Optional

from pymongo.collection import Collection
from pydantic import ValidationError

from core.domain.value_objects.ethical_parameters import EthicalParameters
from application.utils.threshold_scaling_utils import scale_thresholds, exponential_threshold_scaling, linear_threshold_scaling

logger = logging.getLogger(__name__)

class ConfigurationService:
    """Centralized configuration management for ethical evaluation system"""
    
    def __init__(self, db_collection: Optional[Collection] = None):
        """
        Initialize the configuration service.
        
        Args:
            db_collection: MongoDB collection for storing configuration
        """
        self.collection = db_collection
        self.default_parameters = EthicalParameters()
        self._current_parameters = None
        
    @property
    def current_parameters(self) -> EthicalParameters:
        """Get current parameters, loading from DB if needed"""
        if self._current_parameters is None:
            self._load_parameters()
        return self._current_parameters or self.default_parameters
    
    def _load_parameters(self) -> None:
        """Load parameters from database"""
        if self.collection is None:
            self._current_parameters = self.default_parameters
            return
            
        try:
            config_doc = self.collection.find_one({"config_type": "ethical_parameters"})
            if config_doc and "parameters" in config_doc:
                self._current_parameters = EthicalParameters(**config_doc["parameters"])
                logger.info("Loaded parameters from database")
            else:
                self._current_parameters = self.default_parameters
                logger.info("Using default parameters (none found in database)")
        except Exception as e:
            logger.error(f"Error loading parameters: {e}")
            self._current_parameters = self.default_parameters
    
    def save_parameters(self, parameters: EthicalParameters) -> bool:
        """
        Save parameters to database.
        
        Args:
            parameters: Ethical parameters to save
            
        Returns:
            True if successful, False otherwise
        """
        if self.collection is None:
            logger.warning("Cannot save parameters: no database collection")
            return False
            
        try:
            # Validate parameters
            parameters_dict = parameters.model_dump()
            
            result = self.collection.update_one(
                {"config_type": "ethical_parameters"},
                {"$set": {"parameters": parameters_dict}},
                upsert=True
            )
            
            self._current_parameters = parameters
            logger.info(f"Saved parameters to database: {result.modified_count} modified, {result.upserted_id is not None}")
            return True
        except Exception as e:
            logger.error(f"Error saving parameters: {e}")
            return False
    
    def update_parameters(self, parameter_updates: Dict[str, Any]) -> EthicalParameters:
        """
        Update specific parameters.
        
        Args:
            parameter_updates: Dictionary of parameter updates
            
        Returns:
            Updated EthicalParameters
        """
        current = self.current_parameters.model_dump()
        current.update(parameter_updates)
        
        try:
            new_parameters = EthicalParameters(**current)
            self.save_parameters(new_parameters)
            return new_parameters
        except ValidationError as e:
            logger.error(f"Invalid parameter updates: {e}")
            return self.current_parameters
    
    def apply_tau_slider(self, tau_slider: float, scaling_method: str = "exponential") -> EthicalParameters:
        """
        Apply tau slider value to update thresholds.
        
        Args:
            tau_slider: Slider value (0-1)
            scaling_method: Scaling method ("exponential" or "linear")
            
        Returns:
            Updated EthicalParameters
        """
        thresholds = scale_thresholds(tau_slider, scaling_method)
        return self.update_parameters(thresholds)
    
    def get_threshold_distribution(self, num_points: int = 10, scaling_method: str = "exponential") -> Dict[str, Any]:
        """
        Get threshold distribution for visualization.
        
        Args:
            num_points: Number of points to generate
            scaling_method: Scaling method ("exponential" or "linear")
            
        Returns:
            Dictionary with distribution data
        """
        slider_values = [i / (num_points - 1) for i in range(num_points)]
        
        if scaling_method == "exponential":
            threshold_values = [exponential_threshold_scaling(x) for x in slider_values]
        else:
            threshold_values = [linear_threshold_scaling(x) for x in slider_values]
            
        return {
            "slider_values": slider_values,
            "threshold_values": threshold_values,
            "scaling_method": scaling_method
        }
    
    def reset_to_defaults(self) -> EthicalParameters:
        """
        Reset parameters to defaults.
        
        Returns:
            Default EthicalParameters
        """
        self._current_parameters = self.default_parameters
        self.save_parameters(self.default_parameters)
        return self.default_parameters
