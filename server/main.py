"""
FastAPI Backend for Face Verification System

Hỗ trợ 2 luồng chính:
1. Enroll Flow: Upload nhiều ảnh PersonA để tạo gallery
2. Verify Flow: Upload ảnh để kiểm tra có PersonA hay không
"""

import os
import json
import uuid
import shutil
import time
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Any
import tempfile
import logging
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
import uvicorn

# Add src to path to import our modules
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from multi_face_evaluator import MultiFaceEvaluator
from generate_embeddings import EmbeddingGenerator
from inference import load_threshold

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Face Verification API",
    description="API for face enrollment and verification",
    version="1.0.0"
)

# Enable CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Configuration
UPLOAD_DIR = Path("static/uploads")
ANNOTATED_DIR = Path("static/annotated")
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# Ensure directories exist
UPLOAD_DIR.mkdir(exist_ok=True)
ANNOTATED_DIR.mkdir(exist_ok=True)

# Global evaluator (initialized on startup)
evaluator: Optional[MultiFaceEvaluator] = None


class EnrollResponse(BaseModel):
    status: str
    message: str
    num_enrolled: int
    gallery_size: int


class VerifyResponse(BaseModel):
    image: str
    faces: List[Dict[str, Any]]
    detected_count: int
    annotated_image_url: Optional[str]
    processing_time: float
    timestamp: str


class ThresholdResponse(BaseModel):
    threshold: float
    method: str
    accuracy: Optional[float]


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None


@app.on_event("startup")
async def startup_event():
    """Initialize the face verification system on startup."""
    global evaluator
    try:
        logger.info("Initializing Face Verification System...")
        evaluator = MultiFaceEvaluator()
        logger.info("Face Verification System initialized successfully!")
    except Exception as e:
        logger.error(f"Failed to initialize face verification system: {e}")
        raise


def validate_file(file: UploadFile) -> bool:
    """Validate uploaded file."""
    # Check file extension
    if not file.filename:
        return False
    
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        return False
    
    return True


async def save_upload_file(file: UploadFile, destination: Path) -> Path:
    """Save uploaded file to destination."""
    try:
        # Create destination directory if it doesn't exist
        destination.parent.mkdir(parents=True, exist_ok=True)
        
        # Save file
        with destination.open("wb") as buffer:
            content = await file.read()
            if len(content) > MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB"
                )
            buffer.write(content)
        
        return destination
    except Exception as e:
        logger.error(f"Error saving file {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Face Verification API",
        "version": "1.0.0",
        "endpoints": ["/api/enroll", "/api/verify", "/api/threshold"]
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "system_ready": evaluator is not None
    }


@app.get("/api/threshold", response_model=ThresholdResponse)
async def get_threshold():
    """Get current verification threshold."""
    try:
        # Load threshold from config
        config_path = Path("../config/threshold.json")
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            
            return ThresholdResponse(
                threshold=config.get("personA_threshold", 0.6572),
                method=config.get("method", "brute_force_f1"),
                accuracy=config.get("accuracy", None)
            )
        else:
            # Fallback to default
            return ThresholdResponse(
                threshold=0.6572,
                method="default",
                accuracy=None
            )
    except Exception as e:
        logger.error(f"Error getting threshold: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get threshold: {e}")


@app.post("/api/enroll", response_model=EnrollResponse)
async def enroll_person(
    background_tasks: BackgroundTasks,
    images: List[UploadFile] = File(..., description="PersonA images for enrollment")
):
    """Enroll PersonA with uploaded images."""
    try:
        if not images:
            raise HTTPException(status_code=400, detail="No images provided")
        
        # Validate files
        for file in images:
            if not validate_file(file):
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid file: {file.filename}. Allowed: {ALLOWED_EXTENSIONS}"
                )
        
        # Create temporary directory for enrollment
        temp_dir = Path(tempfile.mkdtemp(prefix="enroll_"))
        saved_files = []
        
        try:
            # Save uploaded files
            for i, file in enumerate(images):
                file_ext = Path(file.filename).suffix.lower()
                temp_file = temp_dir / f"enroll_{i:03d}{file_ext}"
                await save_upload_file(file, temp_file)
                saved_files.append(temp_file)
            
            # Generate embeddings for enrolled images
            generator = EmbeddingGenerator()
            
            # Process images and update gallery
            embeddings, filenames = generator.process_folder(str(temp_dir))
            
            if len(embeddings) == 0:
                raise HTTPException(status_code=400, detail="No valid faces found in uploaded images")
            
            # Save updated gallery
            generator.save_embeddings(
                embeddings, 
                filenames,
                output_path="../data/embeddings/personA_normalized.npz"
            )
            
            # Reload evaluator with new gallery
            global evaluator
            evaluator = MultiFaceEvaluator()
            
            # Clean up temporary files in background
            background_tasks.add_task(cleanup_temp_dir, temp_dir)
            
            return EnrollResponse(
                status="success",
                message=f"Successfully enrolled {len(embeddings)} face(s)",
                num_enrolled=len(embeddings),
                gallery_size=len(evaluator.gallery_embeddings)
            )
            
        except Exception as e:
            # Clean up on error
            cleanup_temp_dir(temp_dir)
            raise
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in enrollment: {e}")
        raise HTTPException(status_code=500, detail=f"Enrollment failed: {e}")


@app.post("/api/verify", response_model=VerifyResponse)
async def verify_image(
    file: UploadFile = File(..., description="Image to verify")
):
    """Verify faces in uploaded image."""
    try:
        if not evaluator:
            raise HTTPException(status_code=500, detail="Face verification system not initialized")
        
        # Validate file
        if not validate_file(file):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file: {file.filename}. Allowed: {ALLOWED_EXTENSIONS}"
            )
        
        # Generate unique filename
        file_id = str(uuid.uuid4())
        file_ext = Path(file.filename).suffix.lower()
        temp_filename = f"verify_{file_id}{file_ext}"
        temp_file_path = UPLOAD_DIR / temp_filename
        
        # Save uploaded file
        await save_upload_file(file, temp_file_path)
        
        try:
            # Run verification
            result = evaluator.evaluate_image_multi(str(temp_file_path))
            
            # Generate annotated image if faces were detected
            annotated_url = None
            if result.get('faces') and len(result['faces']) > 0:
                annotated_filename = f"annotated_{file_id}{file_ext}"
                annotated_path = ANNOTATED_DIR / annotated_filename
                
                # Create annotated image
                evaluator.draw_boxes(
                    str(temp_file_path),
                    result['faces'],
                    str(annotated_path)
                )
                
                annotated_url = f"/static/annotated/{annotated_filename}"
            
            return VerifyResponse(
                image=result.get('image', file.filename),
                faces=result.get('faces', []),
                detected_count=result.get('detected_count', 0),
                annotated_image_url=annotated_url,
                processing_time=result.get('processing_time', 0.0),
                timestamp=result.get('timestamp', datetime.now().isoformat())
            )
            
        finally:
            # Clean up uploaded file (keep annotated image)
            if temp_file_path.exists():
                temp_file_path.unlink()
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in verification: {e}")
        raise HTTPException(status_code=500, detail=f"Verification failed: {e}")


@app.delete("/api/cleanup")
async def cleanup_static_files():
    """Clean up old static files."""
    try:
        # Clean up old uploaded files (older than 1 hour)
        from datetime import datetime, timedelta
        
        cutoff_time = datetime.now() - timedelta(hours=1)
        cleaned_count = 0
        
        for file_path in UPLOAD_DIR.glob("*"):
            if file_path.is_file():
                file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                if file_time < cutoff_time:
                    file_path.unlink()
                    cleaned_count += 1
        
        for file_path in ANNOTATED_DIR.glob("*"):
            if file_path.is_file():
                file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                if file_time < cutoff_time:
                    file_path.unlink()
                    cleaned_count += 1
        
        return {"status": "success", "files_cleaned": cleaned_count}
        
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {e}")


def cleanup_temp_dir(temp_dir: Path):
    """Clean up temporary directory."""
    try:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
    except Exception as e:
        logger.warning(f"Failed to clean up temp directory {temp_dir}: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )