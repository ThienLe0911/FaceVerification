#!/usr/bin/env python3
"""
Query Evaluation Module for Face Verification System

This module evaluates hold-out/query images against enrolled PersonA embeddings
using the optimal threshold (0.8466) determined from threshold tuning analysis.

Features:
- Load enrolled PersonA embeddings from NPZ file
- Process query images (positive/negative samples)
- Compute cosine similarity against PersonA mean embedding
- Apply optimal threshold for MATCH/NOT MATCH classification
- Save results in JSON and CSV formats
- Calculate performance metrics (TP, TN, FP, FN, Accuracy)
- Generate comprehensive evaluation report

Author: Face Verification Project
Date: November 2024
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json
import csv
import numpy as np
import torch
from PIL import Image

# Add src directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.inference import FaceVerifier, load_threshold
    FACEVERIFIER_AVAILABLE = True
except ImportError:
    # Fallback to direct FaceNet if inference module not available
    from facenet_pytorch import MTCNN, InceptionResnetV1
    import torch.nn.functional as F
    FACEVERIFIER_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QueryEvaluator:
    """
    Evaluates query images against enrolled PersonA embeddings.
    """
    
    def __init__(
        self,
        embeddings_path: str,
        threshold: Optional[float] = None,
        device: Optional[str] = None
    ):
        """
        Initialize the Query Evaluator.
        
        Args:
            embeddings_path (str): Path to PersonA embeddings NPZ file
            threshold (Optional[float]): Verification threshold. If None, loads from config.
            device (Optional[str]): Device to use ('cuda', 'cpu', 'mps')
        """
        self.embeddings_path = embeddings_path
        
        # Load threshold from config if not provided
        if threshold is None:
            if FACEVERIFIER_AVAILABLE:
                self.threshold = load_threshold()
            else:
                self.threshold = 0.65  # Fallback if config not available
                logger.warning("FaceVerifier not available, using fallback threshold: 0.65")
        else:
            self.threshold = threshold
            logger.info(f"[THRESHOLD] Using provided threshold: {threshold}")
        
        self.device = self._get_device(device)
        
        # Initialize models
        if FACEVERIFIER_AVAILABLE:
            logger.info("Using FaceVerifier from inference module")
            self.face_verifier = FaceVerifier(device=self.device, verification_threshold=threshold)
            self.mtcnn = self.face_verifier.mtcnn
            self.facenet = self.face_verifier.facenet
        else:
            logger.info("Using fallback MTCNN + InceptionResnetV1")
            self._load_fallback_models()
        
        # Load PersonA embeddings
        self.person_a_embeddings = None
        self.person_a_mean = None
        self._load_person_a_embeddings()
        
        logger.info(f"QueryEvaluator initialized with threshold: {threshold}")
    
    def _get_device(self, device: Optional[str] = None) -> str:
        """Determine the best available device."""
        if device is not None:
            return device
        
        if torch.cuda.is_available():
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
    
    def _load_fallback_models(self):
        """Load MTCNN and FaceNet models as fallback."""
        try:
            # Load MTCNN for face detection (CPU to avoid MPS issues)
            self.mtcnn = MTCNN(
                image_size=160,
                margin=0,
                min_face_size=20,
                thresholds=[0.6, 0.7, 0.7],
                factor=0.709,
                post_process=True,
                device='cpu'  # Force CPU for MTCNN
            )
            
            # Load FaceNet model
            self.facenet = InceptionResnetV1(
                pretrained='vggface2'
            ).eval().to(self.device)
            
            logger.info(f"Loaded fallback models: MTCNN (CPU), FaceNet ({self.device})")
            
        except Exception as e:
            logger.error(f"Error loading fallback models: {str(e)}")
            raise
    
    def _load_person_a_embeddings(self):
        """Load PersonA embeddings from NPZ file."""
        try:
            if not os.path.exists(self.embeddings_path):
                raise FileNotFoundError(f"Embeddings not found: {self.embeddings_path}")
            
            # Load embeddings
            data = np.load(self.embeddings_path)
            
            # Try different possible key names
            if 'embeddings' in data:
                self.person_a_embeddings = data['embeddings']  # Shape: (N, 512)
            elif 'per_image' in data:
                self.person_a_embeddings = data['per_image']  # Shape: (N, 512)
            else:
                # Use first array if no known key
                key = list(data.keys())[0]
                self.person_a_embeddings = data[key]
                logger.warning(f"Using key '{key}' for embeddings")
            
            # Check if mean is pre-computed
            if 'mean' in data:
                self.person_a_mean = data['mean']
                logger.info("Using pre-computed PersonA mean embedding")
            else:
                # Compute mean embedding (reference for similarity)
                self.person_a_mean = np.mean(self.person_a_embeddings, axis=0)
                logger.info("Computed PersonA mean from individual embeddings")
            
            # Ensure L2 normalization
            self.person_a_mean = self.person_a_mean / np.linalg.norm(self.person_a_mean)
            
            logger.info(f"Loaded {len(self.person_a_embeddings)} PersonA embeddings")
            logger.info(f"Mean embedding shape: {self.person_a_mean.shape}")
            
        except Exception as e:
            logger.error(f"Error loading PersonA embeddings: {str(e)}")
            raise
    
    def _get_face_embedding(self, image_path: str) -> Optional[np.ndarray]:
        """
        Extract face embedding from image.
        
        Args:
            image_path (str): Path to image file
            
        Returns:
            Optional[np.ndarray]: L2-normalized face embedding or None
        """
        try:
            if FACEVERIFIER_AVAILABLE:
                # Use FaceVerifier
                embedding = self.face_verifier.get_face_embedding(image_path)
                if embedding is not None:
                    # Ensure L2 normalization
                    embedding = embedding / np.linalg.norm(embedding)
                return embedding
            else:
                # Use fallback method
                return self._get_embedding_fallback(image_path)
                
        except Exception as e:
            logger.error(f"Error getting embedding for {image_path}: {str(e)}")
            return None
    
    def _get_embedding_fallback(self, image_path: str) -> Optional[np.ndarray]:
        """Fallback method to get embedding using MTCNN + FaceNet."""
        try:
            # Load and preprocess image
            img = Image.open(image_path).convert('RGB')
            
            # Detect face
            face_tensor = self.mtcnn(img)
            if face_tensor is None:
                logger.warning(f"No face detected in {image_path}")
                return None
            
            # Move to correct device and add batch dimension
            face_tensor = face_tensor.to(self.device).unsqueeze(0)
            
            # Generate embedding
            with torch.no_grad():
                embedding = self.facenet(face_tensor)
                # L2 normalize
                embedding = F.normalize(embedding, p=2, dim=1)
            
            # Convert to numpy
            embedding_np = embedding.cpu().numpy().flatten()
            
            return embedding_np
            
        except Exception as e:
            logger.error(f"Error in fallback embedding for {image_path}: {str(e)}")
            return None
    
    def _compute_cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two L2-normalized embeddings.
        
        Since embeddings are L2-normalized, cosine similarity = dot product.
        """
        return float(np.dot(embedding1, embedding2))
    
    def evaluate_image(self, image_path: str, ground_truth: str) -> Dict:
        """
        Evaluate a single query image.
        
        Args:
            image_path (str): Path to query image
            ground_truth (str): 'positive' or 'negative'
            
        Returns:
            Dict: Evaluation result
        """
        logger.info(f"Evaluating: {image_path}")
        
        # Extract embedding
        embedding = self._get_face_embedding(image_path)
        
        if embedding is None:
            return {
                'filename': os.path.basename(image_path),
                'similarity': 0.0,
                'ground_truth': ground_truth,
                'predicted': 'NOT MATCH',
                'error': 'No face detected'
            }
        
        # Compute similarity with PersonA mean
        similarity = self._compute_cosine_similarity(embedding, self.person_a_mean)
        
        # Make prediction
        predicted = 'MATCH' if similarity >= self.threshold else 'NOT MATCH'
        
        result = {
            'filename': os.path.basename(image_path),
            'similarity': float(similarity),
            'ground_truth': ground_truth,
            'predicted': predicted
        }
        
        logger.info(f"  → Similarity: {similarity:.4f}, Predicted: {predicted}")
        
        return result
    
    def evaluate_query_folder(self, query_dir: str) -> List[Dict]:
        """
        Evaluate all images in query folder structure.
        
        Args:
            query_dir (str): Path to query_images folder
            
        Returns:
            List[Dict]: List of evaluation results
        """
        query_path = Path(query_dir)
        results = []
        
        # Process positive images
        positive_dir = query_path / 'positive'
        if positive_dir.exists():
            logger.info("Processing positive query images...")
            for img_file in positive_dir.iterdir():
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    result = self.evaluate_image(str(img_file), 'positive')
                    results.append(result)
        
        # Process negative images
        negative_dir = query_path / 'negative'
        if negative_dir.exists():
            logger.info("Processing negative query images...")
            for img_file in negative_dir.iterdir():
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    result = self.evaluate_image(str(img_file), 'negative')
                    results.append(result)
        
        logger.info(f"Processed {len(results)} query images")
        return results
    
    def calculate_metrics(self, results: List[Dict]) -> Dict:
        """
        Calculate performance metrics from evaluation results.
        
        Args:
            results (List[Dict]): Evaluation results
            
        Returns:
            Dict: Performance metrics
        """
        # Count outcomes
        tp = fn = fp = tn = 0
        total_positive = total_negative = 0
        match_count = not_match_count = 0
        
        for result in results:
            if 'error' in result:
                continue
                
            ground_truth = result['ground_truth']
            predicted = result['predicted']
            
            # Count by ground truth
            if ground_truth == 'positive':
                total_positive += 1
            else:
                total_negative += 1
            
            # Count by prediction
            if predicted == 'MATCH':
                match_count += 1
            else:
                not_match_count += 1
            
            # Confusion matrix
            if ground_truth == 'positive' and predicted == 'MATCH':
                tp += 1
            elif ground_truth == 'positive' and predicted == 'NOT MATCH':
                fn += 1
            elif ground_truth == 'negative' and predicted == 'MATCH':
                fp += 1
            elif ground_truth == 'negative' and predicted == 'NOT MATCH':
                tn += 1
        
        # Calculate metrics
        total = tp + tn + fp + fn
        accuracy = (tp + tn) / total if total > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'total_images': len(results),
            'total_positive': total_positive,
            'total_negative': total_negative,
            'match_count': match_count,
            'not_match_count': not_match_count,
            'confusion_matrix': {
                'true_positive': tp,
                'true_negative': tn,
                'false_positive': fp,
                'false_negative': fn
            },
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'threshold_used': self.threshold
        }
    
    def save_results(self, results: List[Dict], metrics: Dict, output_dir: str):
        """
        Save evaluation results to JSON and CSV files with metadata logging.
        
        Args:
            results (List[Dict]): Evaluation results
            metrics (Dict): Performance metrics
            output_dir (str): Output directory
        """
        from datetime import datetime
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create metadata for reproducibility
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "threshold_used": self.threshold,
            "embedding_path": self.embeddings_path,
            "model": "FaceNet vggface2",
            "device": self.device,
            "total_query_images": len(results)
        }
        
        # Save JSON results
        json_file = output_path / 'query_results.json'
        json_data = {
            'evaluation_results': results,
            'performance_metrics': metrics,
            'metadata': metadata,
            'configuration': {
                'embeddings_path': self.embeddings_path,
                'threshold': self.threshold,
                'device': self.device
            }
        }
        
        with open(json_file, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        logger.info(f"Saved JSON results to: {json_file}")
        
        # Save CSV results
        csv_file = output_path / 'query_results.csv'
        with open(csv_file, 'w', newline='') as f:
            if results:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
        
        logger.info(f"Saved CSV results to: {csv_file}")
        
        # Save in timestamped runs directory for reproducibility
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        runs_dir = Path("data/evaluation/runs") / timestamp
        runs_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metadata.json
        metadata_file = runs_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save results.csv
        results_csv = runs_dir / "results.csv"
        with open(results_csv, 'w', newline='') as f:
            if results:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
        
        # Save summary.json
        summary_file = runs_dir / "summary.json"
        with open(summary_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Metadata and results saved in runs directory: {runs_dir}")
    
    def print_summary(self, metrics: Dict):
        """Print evaluation summary to console."""
        print("\n" + "="*60)
        print("🎯 QUERY EVALUATION SUMMARY")
        print("="*60)
        
        print(f"📊 Dataset Overview:")
        print(f"   Total Query Images: {metrics['total_images']}")
        print(f"   Positive Samples: {metrics['total_positive']}")
        print(f"   Negative Samples: {metrics['total_negative']}")
        
        print(f"\n🔍 Prediction Results:")
        print(f"   MATCH predictions: {metrics['match_count']}")
        print(f"   NOT MATCH predictions: {metrics['not_match_count']}")
        print(f"   Threshold used: {metrics['threshold_used']:.4f}")
        
        print(f"\n📈 Performance Metrics:")
        print(f"   Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"   Precision: {metrics['precision']:.4f}")
        print(f"   Recall: {metrics['recall']:.4f}")
        print(f"   F1-Score: {metrics['f1_score']:.4f}")
        
        cm = metrics['confusion_matrix']
        print(f"\n📋 Confusion Matrix:")
        print(f"   True Positive (TP):  {cm['true_positive']:2d} (PersonA correctly identified)")
        print(f"   True Negative (TN):  {cm['true_negative']:2d} (Others correctly rejected)")
        print(f"   False Positive (FP): {cm['false_positive']:2d} (Others incorrectly matched)")
        print(f"   False Negative (FN): {cm['false_negative']:2d} (PersonA incorrectly rejected)")
        
        # Classification report style
        print(f"\n📊 Detailed Results:")
        print(f"   Sensitivity (TPR):   {cm['true_positive']}/{cm['true_positive'] + cm['false_negative']} = {metrics['recall']:.4f}")
        print(f"   Specificity (TNR):   {cm['true_negative']}/{cm['true_negative'] + cm['false_positive']} = {cm['true_negative']/(cm['true_negative'] + cm['false_positive']) if (cm['true_negative'] + cm['false_positive']) > 0 else 0:.4f}")
        
        print("="*60)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Evaluate query images against PersonA embeddings")
    
    parser.add_argument(
        '--query_dir',
        type=str,
        default='data/processed/query_images',
        help='Directory containing query images (default: data/processed/query_images)'
    )
    
    parser.add_argument(
        '--emb_path',
        type=str,
        default='data/embeddings/personA_normalized.npz',
        help='Path to PersonA embeddings NPZ file (default: data/embeddings/personA_normalized.npz)'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        default=None,
        help='Verification threshold (default: load from config/threshold.json)'
    )
    
    parser.add_argument(
        '--out_dir',
        type=str,
        default='data/evaluation',
        help='Output directory for results (default: data/evaluation)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['cpu', 'cuda', 'mps'],
        help='Device to use (auto-detect if not specified)'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize evaluator
        logger.info("Initializing Query Evaluator...")
        evaluator = QueryEvaluator(
            embeddings_path=args.emb_path,
            threshold=args.threshold,
            device=args.device
        )
        
        # Evaluate query images
        logger.info(f"Evaluating query images from: {args.query_dir}")
        results = evaluator.evaluate_query_folder(args.query_dir)
        
        if not results:
            logger.error("No query images found or processed!")
            return
        
        # Calculate metrics
        metrics = evaluator.calculate_metrics(results)
        
        # Save results
        evaluator.save_results(results, metrics, args.out_dir)
        
        # Print summary
        evaluator.print_summary(metrics)
        
        logger.info("Query evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        raise


if __name__ == "__main__":
    main()