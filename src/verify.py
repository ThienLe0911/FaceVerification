#!/usr/bin/env python3
"""
CLI Verify Module for Face Verification System

This module provides a command-line interface for verifying single images
against enrolled PersonA embeddings using the optimal threshold.

Features:
- Load PersonA embeddings from NPZ file
- Process single input image
- Compute cosine similarity against PersonA mean embedding
- Apply threshold for MATCH/NOT MATCH classification
- Save verification result to JSON log
- Print detailed verification report

Usage:
    python src/verify.py --image path/to/image.jpg
    python src/verify.py --image path/to/image.jpg --threshold 0.7

Author: Face Verification Project
Date: November 2024
"""

import os
import sys
import argparse
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
import numpy as np

# Add src directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference import FaceVerifier, load_threshold

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SingleImageVerifier:
    """
    Handles verification of single images against PersonA embeddings.
    """
    
    def __init__(
        self,
        embeddings_path: str = "data/embeddings/personA_normalized.npz",
        threshold: Optional[float] = None
    ):
        """
        Initialize the Single Image Verifier.
        
        Args:
            embeddings_path (str): Path to PersonA embeddings NPZ file
            threshold (Optional[float]): Verification threshold. If None, loads from config.
        """
        self.embeddings_path = embeddings_path
        
        # Load threshold
        if threshold is not None:
            self.threshold = threshold
            logger.info(f"[THRESHOLD] Using provided threshold: {threshold}")
        else:
            self.threshold = load_threshold()
        
        # Initialize FaceVerifier
        self.face_verifier = FaceVerifier(verification_threshold=self.threshold)
        
        # Load PersonA embeddings
        self.person_a_mean = self._load_person_a_embeddings()
        
        logger.info("SingleImageVerifier initialized successfully")
    
    def _load_person_a_embeddings(self) -> np.ndarray:
        """Load PersonA mean embedding from NPZ file."""
        try:
            if not os.path.exists(self.embeddings_path):
                raise FileNotFoundError(f"PersonA embeddings not found: {self.embeddings_path}")
            
            # Load embeddings
            data = np.load(self.embeddings_path)
            
            # Try different possible key names
            if 'per_image' in data:
                embeddings = data['per_image']
            elif 'embeddings' in data:
                embeddings = data['embeddings']
            else:
                # Use first array if no known key
                key = list(data.keys())[0]
                embeddings = data[key]
                logger.warning(f"Using key '{key}' for embeddings")
            
            # Check if mean is pre-computed
            if 'mean' in data:
                person_a_mean = data['mean']
                logger.info("Using pre-computed PersonA mean embedding")
            else:
                # Compute mean embedding
                person_a_mean = np.mean(embeddings, axis=0)
                logger.info("Computed PersonA mean from individual embeddings")
            
            # Ensure L2 normalization
            person_a_mean = person_a_mean / np.linalg.norm(person_a_mean)
            
            logger.info(f"Loaded PersonA embeddings: {len(embeddings)} samples")
            logger.info(f"Mean embedding shape: {person_a_mean.shape}")
            
            return person_a_mean
            
        except Exception as e:
            logger.error(f"Error loading PersonA embeddings: {str(e)}")
            raise
    
    def _compute_cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two L2-normalized embeddings.
        Since embeddings are L2-normalized, cosine similarity = dot product.
        """
        return float(np.dot(embedding1, embedding2))
    
    def verify_image(self, image_path: str) -> Dict:
        """
        Verify a single image against PersonA embeddings.
        
        Args:
            image_path (str): Path to input image
            
        Returns:
            Dict: Verification result with metadata
        """
        logger.info(f"Verifying image: {image_path}")
        
        # Extract embedding from input image
        embedding = self.face_verifier.get_face_embedding(image_path)
        
        if embedding is None:
            result = {
                "image_path": image_path,
                "filename": os.path.basename(image_path),
                "similarity": 0.0,
                "threshold_used": self.threshold,
                "prediction": "NOT MATCH",
                "error": "No face detected",
                "timestamp": datetime.now().isoformat(),
                "embeddings_path": self.embeddings_path,
                "device": self.face_verifier.device
            }
            logger.warning("No face detected in input image")
            return result
        
        # Compute similarity with PersonA mean
        similarity = self._compute_cosine_similarity(embedding, self.person_a_mean)
        
        # Make prediction
        prediction = "MATCH" if similarity >= self.threshold else "NOT MATCH"
        
        result = {
            "image_path": image_path,
            "filename": os.path.basename(image_path),
            "similarity": float(similarity),
            "threshold_used": self.threshold,
            "prediction": prediction,
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "threshold_used": self.threshold,
                "embedding_path": self.embeddings_path,
                "model": "FaceNet vggface2",
                "device": self.face_verifier.device,
                "query_image": os.path.basename(image_path)
            },
            "timestamp": datetime.now().isoformat(),
            "embeddings_path": self.embeddings_path,
            "device": self.face_verifier.device
        }
        
        logger.info(f"Verification completed: {prediction} (similarity: {similarity:.4f})")
        
        return result
    
    def save_verification_log(self, result: Dict, log_dir: str = "data/verification_logs"):
        """
        Save verification result to JSON log with timestamped runs.
        
        Args:
            result (Dict): Verification result
            log_dir (str): Directory to save logs
        """
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        # Save as latest_verify.json
        latest_file = log_path / "latest_verify.json"
        with open(latest_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        # Also save with timestamp for history
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        timestamped_file = log_path / f"verify_{timestamp}.json"
        with open(timestamped_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        # Save in evaluation/runs directory for reproducibility
        runs_dir = Path("data/evaluation/runs") / timestamp
        runs_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metadata.json
        metadata_file = runs_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(result["metadata"], f, indent=2)
        
        # Save complete result
        result_file = runs_dir / "verify_result.json"
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        logger.info(f"Verification log saved: {latest_file}")
        logger.info(f"Timestamped log saved: {timestamped_file}")
        logger.info(f"Metadata saved in runs: {runs_dir}")
    
    
    def print_verification_report(self, result: Dict):
        """Print detailed verification report to console."""
        print("\n" + "="*60)
        print("🔍 FACE VERIFICATION REPORT")
        print("="*60)
        
        print(f"📁 Input Image: {result['filename']}")
        print(f"📊 Similarity Score: {result['similarity']:.4f}")
        print(f"🎯 Threshold Used: {result['threshold_used']:.4f}")
        print(f"🚦 Verification Result: {result['prediction']}")
        
        if 'error' in result:
            print(f"⚠️  Error: {result['error']}")
        
        print(f"⏰ Timestamp: {result['timestamp']}")
        print(f"🖥️  Device: {result['device']}")
        print(f"📂 PersonA Embeddings: {result['embeddings_path']}")
        
        # Visual indicator
        if result['prediction'] == "MATCH":
            print("\n✅ MATCH - This person is identified as PersonA")
        else:
            print("\n❌ NOT MATCH - This person is not PersonA")
        
        print("="*60)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Verify a single image against PersonA embeddings")
    
    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='Path to input image for verification'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        default=None,
        help='Verification threshold override (default: load from config)'
    )
    
    parser.add_argument(
        '--emb_path',
        type=str,
        default='data/embeddings/personA_normalized.npz',
        help='Path to PersonA embeddings (default: data/embeddings/personA_normalized.npz)'
    )
    
    parser.add_argument(
        '--log_dir',
        type=str,
        default='data/verification_logs',
        help='Directory to save verification logs (default: data/verification_logs)'
    )
    
    args = parser.parse_args()
    
    try:
        # Validate input image
        if not os.path.exists(args.image):
            raise FileNotFoundError(f"Input image not found: {args.image}")
        
        # Initialize verifier
        logger.info("Initializing Single Image Verifier...")
        verifier = SingleImageVerifier(
            embeddings_path=args.emb_path,
            threshold=args.threshold
        )
        
        # Perform verification
        result = verifier.verify_image(args.image)
        
        # Save log
        verifier.save_verification_log(result, args.log_dir)
        
        # Print report
        verifier.print_verification_report(result)
        
        logger.info("Single image verification completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during verification: {str(e)}")
        raise


if __name__ == "__main__":
    main()