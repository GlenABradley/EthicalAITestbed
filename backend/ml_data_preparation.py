"""
ML DATA PREPARATION MODULE

This module handles the preparation of ML training data by:
- Processing uploaded text files through the ethical evaluation engine
- Tagging content with ethical and intent vectors
- Formatting output for ML training

Core capabilities:
- Multi-format input support (plain text, JSON, JSONL)
- Ethical vector and intent vector tagging
- Standardized ML-ready output generation
"""

import json
import os
import uuid
from pathlib import Path
from typing import Dict, List, Any, Union, Optional

from fastapi import UploadFile
import torch
import numpy as np

from ethical_engine import EthicalEvaluator, IntentHierarchy


class MLDataPreparationService:
    """
    Service for preparing machine learning training data with ethical and intent vectors.
    
    This service processes text data through the ethical evaluation engine
    and generates ML-ready output with comprehensive ethical tagging.
    """

    def __init__(self, ethical_evaluator: EthicalEvaluator):
        """
        Initialize the ML data preparation service.
        
        Args:
            ethical_evaluator: Instance of the EthicalEvaluator for text analysis
        """
        self.ethical_evaluator = ethical_evaluator
        self.upload_dir = Path("./uploads")
        self.output_dir = Path("./ml_outputs")
        
        # Create directories if they don't exist
        self.upload_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)

    async def process_file(self, file: UploadFile) -> Dict[str, Any]:
        """
        Process an uploaded file through the ethical evaluation pipeline.
        
        Args:
            file: The uploaded file to process
            
        Returns:
            Dictionary with processing results, including output file path
        """
        # Generate unique IDs for the upload and output
        file_id = str(uuid.uuid4())
        input_path = self.upload_dir / f"{file_id}_{file.filename}"
        output_path = self.output_dir / f"{file_id}_ethical_vectors.jsonl"
        
        # Save uploaded file
        with open(input_path, "wb") as f:
            content = await file.read()
            f.write(content)
            
        # Determine file type and parse content
        file_ext = input_path.suffix.lower()
        content_items = self._parse_file(input_path, file_ext)
        
        # Process content through ethical engine
        processed_items = []
        for item in content_items:
            processed_item = await self._process_text_item(item)
            processed_items.append(processed_item)
            
        # Write output to JSONL file
        with open(output_path, "w") as f:
            for item in processed_items:
                f.write(json.dumps(item) + "\n")
        
        return {
            "status": "success",
            "input_file": file.filename,
            "output_file": output_path.name,
            "items_processed": len(processed_items),
            "output_path": str(output_path),
        }
    
    def _parse_file(self, file_path: Path, file_ext: str) -> List[Dict[str, Any]]:
        """
        Parse the input file based on its format.
        
        Args:
            file_path: Path to the file
            file_ext: File extension
            
        Returns:
            List of text items for processing
        """
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        if file_ext == ".json":
            # Handle JSON file
            try:
                data = json.loads(content)
                if isinstance(data, list):
                    return data
                else:
                    return [data]
            except json.JSONDecodeError:
                raise ValueError(f"Invalid JSON format in {file_path}")
                
        elif file_ext == ".jsonl":
            # Handle JSONL file
            items = []
            for line in content.strip().split("\n"):
                if line.strip():
                    try:
                        items.append(json.loads(line))
                    except json.JSONDecodeError:
                        raise ValueError(f"Invalid JSON line in {file_path}: {line}")
            return items
            
        else:
            # Handle plain text file - split by paragraphs
            paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
            return [{"text": p} for p in paragraphs]
    
    async def _process_text_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single text item through the ethical evaluation engine.
        
        Args:
            item: Dictionary containing text data
            
        Returns:
            Original item enriched with ethical and intent vectors
        """
        # Extract text from the item (could be at root or in a 'text' field)
        text = item.get("text", "") if isinstance(item, dict) else str(item)
        
        # Skip empty text
        if not text.strip():
            return item
            
        # Perform ethical evaluation
        evaluation = self.ethical_evaluator.evaluate_text(text)
        
        # Extract intent vectors
        intent_hierarchy = IntentHierarchy()
        intent_vectors = intent_hierarchy.classify_intent(text)
        
        # Create a new item with the original content and enriched data
        enriched_item = item.copy() if isinstance(item, dict) else {"text": text}
        
        # Add ethical vectors
        enriched_item["ethical_vectors"] = {
            "virtue": self._process_perspective_vector(evaluation.virtue_perspective),
            "deontological": self._process_perspective_vector(evaluation.deontological_perspective),
            "consequentialist": self._process_perspective_vector(evaluation.consequentialist_perspective),
        }
        
        # Add intent vectors
        enriched_item["intent_vectors"] = intent_vectors
        
        # Add aggregate metrics
        enriched_item["ethical_metrics"] = {
            "overall_score": evaluation.aggregate_score,
            "has_violations": evaluation.has_violations,
            "certainty": evaluation.certainty
        }
        
        return enriched_item
    
    def _process_perspective_vector(self, perspective) -> Dict[str, Any]:
        """
        Process a perspective vector from the evaluation into a serializable format.
        
        Args:
            perspective: The perspective object from ethical evaluation
            
        Returns:
            Dictionary with vector data in serializable format
        """
        if hasattr(perspective, "projection_values") and perspective.projection_values is not None:
            if isinstance(perspective.projection_values, torch.Tensor):
                projection_values = perspective.projection_values.cpu().numpy().tolist()
            elif isinstance(perspective.projection_values, np.ndarray):
                projection_values = perspective.projection_values.tolist()
            else:
                projection_values = perspective.projection_values
        else:
            projection_values = []
            
        return {
            "score": getattr(perspective, "score", 0.0),
            "projection_values": projection_values,
            "violations": getattr(perspective, "violations", []),
            "analysis": getattr(perspective, "analysis", "")
        }
