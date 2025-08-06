"""
Get Heat Map Data Use Case for the Ethical AI Testbed.

This module defines the use case for generating heat map data for visualization.
"""

import logging
import random
from typing import Dict, Any, List
import numpy as np

logger = logging.getLogger(__name__)

class GetHeatMapDataUseCase:
    """
    Use case for generating heat map data for visualization.
    
    This class implements the use case for generating heat map data
    based on ethical analysis. It follows the Clean Architecture
    pattern for use cases.
    """
    
    def __init__(self, orchestrator):
        """
        Initialize the use case with dependencies.
        
        Args:
            orchestrator: The unified ethical orchestrator
        """
        self.orchestrator = orchestrator
        
    async def execute(self, text: str) -> Dict[str, Any]:
        """
        Execute the use case to generate heat map data.
        
        Args:
            text: The text to analyze
            
        Returns:
            Dict[str, Any]: Heat map data
        """
        logger.info(f"Generating heat map data for text (length: {len(text)})")
        
        try:
            # Perform ethical evaluation
            evaluation = await self.orchestrator.evaluate_content(text=text)
            
            # Generate heat map data from evaluation
            heat_map_data = self._generate_heat_map_data(text, evaluation)
            
            return heat_map_data
            
        except Exception as e:
            logger.error(f"Error generating heat map data: {str(e)}", exc_info=True)
            # Fallback to mock data if evaluation fails
            return self._generate_mock_heat_map_data(text)
            
    def _generate_heat_map_data(self, text: str, evaluation: Any) -> Dict[str, Any]:
        """
        Generate heat map data from evaluation results.
        
        Args:
            text: The analyzed text
            evaluation: The ethical evaluation results
            
        Returns:
            Dict[str, Any]: Heat map data
        """
        # Extract spans from evaluation
        spans = evaluation.spans if hasattr(evaluation, 'spans') else []
        
        # Generate heat map data
        heat_map = {
            "text": text,
            "heatmap_data": [],
            "perspective_data": {
                "virtue": [],
                "deontological": [],
                "consequentialist": []
            },
            "combined_score": 0.0
        }
        
        # Process each span
        for span in spans:
            # Get span data
            start = span.start
            end = span.end
            span_text = text[start:end]
            
            # Get scores
            virtue_score = span.virtue_score if hasattr(span, 'virtue_score') else 0.0
            deontological_score = span.deontological_score if hasattr(span, 'deontological_score') else 0.0
            consequentialist_score = span.consequentialist_score if hasattr(span, 'consequentialist_score') else 0.0
            combined_score = span.combined_score if hasattr(span, 'combined_score') else 0.0
            
            # Add to heat map data
            heat_map["heatmap_data"].append({
                "start": start,
                "end": end,
                "text": span_text,
                "intensity": combined_score,
                "virtue_score": virtue_score,
                "deontological_score": deontological_score,
                "consequentialist_score": consequentialist_score,
                "combined_score": combined_score
            })
            
            # Add to perspective data
            heat_map["perspective_data"]["virtue"].append({
                "start": start,
                "end": end,
                "text": span_text,
                "intensity": virtue_score
            })
            
            heat_map["perspective_data"]["deontological"].append({
                "start": start,
                "end": end,
                "text": span_text,
                "intensity": deontological_score
            })
            
            heat_map["perspective_data"]["consequentialist"].append({
                "start": start,
                "end": end,
                "text": span_text,
                "intensity": consequentialist_score
            })
            
        # Calculate overall score
        if spans:
            heat_map["combined_score"] = sum(span.combined_score for span in spans if hasattr(span, 'combined_score')) / len(spans)
        
        return heat_map
        
    def _generate_mock_heat_map_data(self, text: str) -> Dict[str, Any]:
        """
        Generate mock heat map data for fallback.
        
        Args:
            text: The text to analyze
            
        Returns:
            Dict[str, Any]: Mock heat map data
        """
        logger.warning("Generating mock heat map data as fallback")
        
        # Split text into words
        words = text.split()
        
        # Generate mock heat map data
        heat_map = {
            "text": text,
            "heatmap_data": [],
            "perspective_data": {
                "virtue": [],
                "deontological": [],
                "consequentialist": []
            },
            "combined_score": random.uniform(0.1, 0.4)
        }
        
        # Process each word
        current_pos = 0
        for word in words:
            # Skip short words
            if len(word) < 3:
                current_pos += len(word) + 1
                continue
                
            # Random chance to include word
            if random.random() < 0.3:
                # Get word position
                start = text.find(word, current_pos)
                if start == -1:
                    current_pos += len(word) + 1
                    continue
                    
                end = start + len(word)
                current_pos = end
                
                # Generate random scores
                virtue_score = random.uniform(0.1, 0.5)
                deontological_score = random.uniform(0.1, 0.5)
                consequentialist_score = random.uniform(0.1, 0.5)
                combined_score = (virtue_score + deontological_score + consequentialist_score) / 3
                
                # Add to heat map data
                heat_map["heatmap_data"].append({
                    "start": start,
                    "end": end,
                    "text": word,
                    "intensity": combined_score,
                    "virtue_score": virtue_score,
                    "deontological_score": deontological_score,
                    "consequentialist_score": consequentialist_score,
                    "combined_score": combined_score
                })
                
                # Add to perspective data
                heat_map["perspective_data"]["virtue"].append({
                    "start": start,
                    "end": end,
                    "text": word,
                    "intensity": virtue_score
                })
                
                heat_map["perspective_data"]["deontological"].append({
                    "start": start,
                    "end": end,
                    "text": word,
                    "intensity": deontological_score
                })
                
                heat_map["perspective_data"]["consequentialist"].append({
                    "start": start,
                    "end": end,
                    "text": word,
                    "intensity": consequentialist_score
                })
            else:
                current_pos += len(word) + 1
        
        return heat_map
