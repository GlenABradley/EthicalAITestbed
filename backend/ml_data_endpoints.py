"""
ML DATA PREPARATION ENDPOINTS

API endpoints for the ML data preparation feature, including:
- File upload and processing
- Ethical vector tagging
- Intent vector generation
- ML-ready output generation
"""

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from typing import Dict, Any, List, Optional
import os
import json
from pathlib import Path

from ethical_engine import EthicalEvaluator
from ml_data_preparation import MLDataPreparationService

# Avoid circular import
_global_ethical_engine = None

def get_ml_ethical_engine() -> EthicalEvaluator:
    """Get or create the ethical engine instance."""
    global _global_ethical_engine
    if _global_ethical_engine is None:
        _global_ethical_engine = EthicalEvaluator()
    return _global_ethical_engine

# Create API router
router = APIRouter(
    prefix="/api/ml-data",
    tags=["ML Data Preparation"],
    responses={404: {"description": "Not found"}},
)

# Create a data directory if it doesn't exist
OUTPUT_DIR = Path("./ml_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


async def get_ml_data_service() -> MLDataPreparationService:
    """Dependency to get the ML data preparation service."""
    evaluator = get_ml_ethical_engine()
    return MLDataPreparationService(evaluator)


@router.post("/upload-process")
async def upload_and_process_file(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
    data_service: MLDataPreparationService = Depends(get_ml_data_service),
) -> Dict[str, Any]:
    """
    Upload and process a text file through the ethical evaluation pipeline.
    
    The file will be analyzed and enriched with ethical vectors and intent vectors,
    producing ML-ready training data output in JSONL format.
    
    Args:
        file: The uploaded file (text, JSON, or JSONL format)
        background_tasks: Background tasks manager
        data_service: ML data preparation service instance
        
    Returns:
        Dictionary with processing results and output file information
    """
    # Validate file extension
    allowed_extensions = ['.txt', '.json', '.jsonl']
    file_ext = os.path.splitext(file.filename)[1].lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file format. Allowed formats: {', '.join(allowed_extensions)}"
        )
    
    # Process the file
    try:
        result = await data_service.process_file(file)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@router.get("/download/{filename}")
async def download_processed_file(filename: str) -> FileResponse:
    """
    Download a processed ML data file.
    
    Args:
        filename: The name of the file to download
        
    Returns:
        The file as a downloadable response
    """
    file_path = OUTPUT_DIR / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="application/octet-stream"
    )


@router.get("/list-outputs")
async def list_processed_files() -> List[Dict[str, Any]]:
    """
    List all available processed ML data files.
    
    Returns:
        List of file information dictionaries
    """
    files = []
    
    if OUTPUT_DIR.exists():
        for file_path in OUTPUT_DIR.iterdir():
            if file_path.is_file() and file_path.suffix == '.jsonl':
                files.append({
                    "filename": file_path.name,
                    "size_bytes": file_path.stat().st_size,
                    "created_at": file_path.stat().st_ctime,
                })
    
    return files


@router.get("/sample-output")
async def get_sample_output() -> Dict[str, Any]:
    """
    Get a sample of what the ML data preparation output looks like.
    This is useful for documentation and UI examples.
    
    Returns:
        Dictionary containing a sample output structure
    """
    return {
        "text": "This is an example text that would be analyzed for ethical content.",
        "ethical_vectors": {
            "virtue": {
                "score": 0.92,
                "projection_values": [0.87, 0.93, 0.94, 0.89, 0.95],
                "violations": [],
                "analysis": "Text adheres to virtue ethics principles."
            },
            "deontological": {
                "score": 0.88,
                "projection_values": [0.89, 0.87, 0.82, 0.91, 0.88],
                "violations": [],
                "analysis": "Text follows deontological ethical guidelines."
            },
            "consequentialist": {
                "score": 0.91,
                "projection_values": [0.92, 0.89, 0.93, 0.90, 0.91],
                "violations": [],
                "analysis": "Text has positive consequentialist outcomes."
            }
        },
        "intent_vectors": {
            "manipulation": 0.02,
            "deception": 0.01,
            "harm": 0.01,
            "coercion": 0.01,
            "fraud": 0.01,
            "information": 0.92,
            "assistance": 0.89,
            "education": 0.94
        },
        "ethical_metrics": {
            "overall_score": 0.90,
            "has_violations": False,
            "certainty": 0.95
        }
    }
