#!/usr/bin/env python3
"""
Extended Evaluation Module for Face Verification System

This module provides scalable evaluation capabilities for the face verification system,
allowing reproducible sampling of large negative datasets and comprehensive analysis.

Features:
- Reproducible negative sampling from large pools
- Extended positive sample integration
- Run-based artifact management with timestamping
- Comprehensive visualization and analysis
- Progress tracking and detailed logging
- Performance validation with configurable thresholds

Author: Face Verification Project
Date: November 2024
"""

import os
import sys
import argparse
import logging
import json
import csv
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
import random
from tqdm import tqdm
import torch
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

# Add src directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.inference import FaceVerifier, load_threshold
    from src.evaluate_queries import QueryEvaluator
    INFERENCE_AVAILABLE = True
except ImportError:
    # Fallback imports
    from facenet_pytorch import MTCNN, InceptionResnetV1
    import torch.nn.functional as F
    INFERENCE_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ExtendedEvaluator:
    """
    Extended evaluation system for face verification with reproducible sampling.
    """
    
    def __init__(
        self,
        emb_path: str,
        threshold_config: str,
        seed: int = 42
    ):
        """
        Initialize the Extended Evaluator.
        
        Args:
            emb_path (str): Path to PersonA embeddings NPZ file
            threshold_config (str): Path to threshold configuration JSON
            seed (int): Random seed for reproducible sampling
        """
        self.emb_path = emb_path
        self.threshold_config = threshold_config
        self.seed = seed
        
        # Set random seeds for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Load threshold from config
        self.threshold = self._load_threshold()
        
        # Initialize models
        if INFERENCE_AVAILABLE:
            logger.info("Using FaceVerifier from inference module")
            self.face_verifier = FaceVerifier(verification_threshold=self.threshold)
        else:
            logger.info("Using fallback MTCNN + InceptionResnetV1")
            self._init_fallback_models()
        
        # Load PersonA embeddings
        self.person_a_mean = self._load_person_a_embeddings()
        
        logger.info(f"ExtendedEvaluator initialized with threshold: {self.threshold}")
    
    def _load_threshold(self) -> float:
        """Load threshold from configuration file."""
        try:
            if INFERENCE_AVAILABLE:
                return load_threshold(self.threshold_config)
            else:
                # Fallback threshold loading
                if os.path.exists(self.threshold_config):
                    with open(self.threshold_config, 'r') as f:
                        config = json.load(f)
                    threshold = config.get("personA_threshold", 0.65)
                    logger.info(f"[THRESHOLD] Loaded threshold: {threshold}")
                    return threshold
                else:
                    logger.warning(f"Threshold config not found, using fallback: 0.65")
                    return 0.65
        except Exception as e:
            logger.error(f"Error loading threshold: {e}, using fallback: 0.65")
            return 0.65
    
    def _init_fallback_models(self):
        """Initialize fallback models if inference module unavailable."""
        try:
            # MTCNN for face detection
            self.mtcnn = MTCNN(
                image_size=160,
                margin=0,
                min_face_size=20,
                thresholds=[0.6, 0.7, 0.7],
                factor=0.709,
                post_process=True,
                device='cpu'
            )
            
            # FaceNet model
            self.facenet = InceptionResnetV1(
                pretrained='vggface2'
            ).eval()
            
            # Device detection
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
            
            self.facenet = self.facenet.to(self.device)
            
            logger.info(f"Initialized fallback models on device: {self.device}")
            
        except Exception as e:
            logger.error(f"Error initializing fallback models: {e}")
            raise
    
    def _load_person_a_embeddings(self) -> np.ndarray:
        """Load PersonA mean embedding from NPZ file."""
        try:
            if not os.path.exists(self.emb_path):
                raise FileNotFoundError(f"PersonA embeddings not found: {self.emb_path}")
            
            data = np.load(self.emb_path)
            
            # Try different key names
            if 'per_image' in data:
                embeddings = data['per_image']
            elif 'embeddings' in data:
                embeddings = data['embeddings']
            else:
                key = list(data.keys())[0]
                embeddings = data[key]
                logger.warning(f"Using key '{key}' for embeddings")
            
            # Use pre-computed mean or compute it
            if 'mean' in data:
                person_a_mean = data['mean']
                logger.info("Using pre-computed PersonA mean embedding")
            else:
                person_a_mean = np.mean(embeddings, axis=0)
                logger.info("Computed PersonA mean from individual embeddings")
            
            # Ensure L2 normalization
            person_a_mean = person_a_mean / np.linalg.norm(person_a_mean)
            
            logger.info(f"Loaded PersonA embeddings: {len(embeddings)} samples")
            return person_a_mean
            
        except Exception as e:
            logger.error(f"Error loading PersonA embeddings: {e}")
            raise
    
    def sample_negatives(self, others_dir: str, num_neg: int) -> List[str]:
        """
        Sample negative images reproducibly from others directory.
        
        Args:
            others_dir (str): Directory containing negative sample pool
            num_neg (int): Number of negatives to sample
            
        Returns:
            List[str]: List of sampled negative image paths
        """
        others_path = Path(others_dir)
        
        if not others_path.exists():
            raise FileNotFoundError(f"Others directory not found: {others_dir}")
        
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        all_images = []
        
        for ext in image_extensions:
            all_images.extend(list(others_path.glob(f'*{ext}')))
            all_images.extend(list(others_path.glob(f'*{ext.upper()}')))
        
        if len(all_images) < num_neg:
            logger.warning(f"Requested {num_neg} negatives but only {len(all_images)} available")
            num_neg = len(all_images)
        
        # Sort for reproducibility across different systems
        all_images = sorted(all_images)
        
        # Sample reproducibly
        sampled_images = random.sample(all_images, num_neg)
        
        logger.info(f"Sampled {len(sampled_images)} negatives from {len(all_images)} total")
        
        return [str(img) for img in sampled_images]
    
    def create_run_folder(self, out_root: str) -> Path:
        """
        Create timestamped run folder for this evaluation.
        
        Args:
            out_root (str): Root directory for runs
            
        Returns:
            Path: Created run folder path
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_folder = Path(out_root) / timestamp
        run_folder.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Created run folder: {run_folder}")
        return run_folder
    
    def setup_run_images(
        self,
        run_folder: Path,
        query_dir: str,
        sampled_negatives: List[str],
        add_pos_dir: Optional[str] = None
    ) -> Tuple[List[str], List[str]]:
        """
        Setup images for this run by copying/linking to run folder.
        
        Args:
            run_folder (Path): Run folder path
            query_dir (str): Original query images directory
            sampled_negatives (List[str]): Sampled negative image paths
            add_pos_dir (Optional[str]): Additional positive images directory
            
        Returns:
            Tuple[List[str], List[str]]: (positive_images, negative_images)
        """
        # Create run query structure
        run_query_dir = run_folder / "query_images"
        run_positive_dir = run_query_dir / "positive"
        run_negative_dir = run_query_dir / "negative"
        
        run_positive_dir.mkdir(parents=True, exist_ok=True)
        run_negative_dir.mkdir(parents=True, exist_ok=True)
        
        positive_images = []
        negative_images = []
        
        # Copy existing positive images
        original_positive_dir = Path(query_dir) / "positive"
        if original_positive_dir.exists():
            for img_file in original_positive_dir.iterdir():
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    dst = run_positive_dir / img_file.name
                    shutil.copy2(img_file, dst)
                    positive_images.append(str(dst))
        
        # Add additional positives if provided
        if add_pos_dir and os.path.exists(add_pos_dir):
            add_pos_path = Path(add_pos_dir)
            for img_file in add_pos_path.iterdir():
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    dst = run_positive_dir / f"extra_{img_file.name}"
                    shutil.copy2(img_file, dst)
                    positive_images.append(str(dst))
            logger.info(f"Added {len(list(add_pos_path.iterdir()))} additional positives")
        
        # Copy existing negative images
        original_negative_dir = Path(query_dir) / "negative"
        if original_negative_dir.exists():
            for img_file in original_negative_dir.iterdir():
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    dst = run_negative_dir / img_file.name
                    shutil.copy2(img_file, dst)
                    negative_images.append(str(dst))
        
        # Copy sampled negatives
        for i, neg_path in enumerate(sampled_negatives):
            src_path = Path(neg_path)
            dst = run_negative_dir / f"sampled_{i:04d}_{src_path.name}"
            shutil.copy2(src_path, dst)
            negative_images.append(str(dst))
        
        logger.info(f"Setup run images: {len(positive_images)} positives, {len(negative_images)} negatives")
        
        return positive_images, negative_images
    
    def get_face_embedding(self, image_path: str) -> Tuple[Optional[np.ndarray], bool]:
        """
        Extract face embedding from image.
        
        Args:
            image_path (str): Path to image file
            
        Returns:
            Tuple[Optional[np.ndarray], bool]: (embedding, detection_success)
        """
        try:
            if INFERENCE_AVAILABLE:
                embedding = self.face_verifier.get_face_embedding(image_path)
                detected = embedding is not None
                if detected:
                    embedding = embedding / np.linalg.norm(embedding)  # Ensure L2 norm
                return embedding, detected
            else:
                return self._get_embedding_fallback(image_path)
        except Exception as e:
            logger.error(f"Error getting embedding for {image_path}: {e}")
            return None, False
    
    def _get_embedding_fallback(self, image_path: str) -> Tuple[Optional[np.ndarray], bool]:
        """Fallback method to get embedding."""
        try:
            img = Image.open(image_path).convert('RGB')
            face_tensor = self.mtcnn(img)
            
            if face_tensor is None:
                return None, False
            
            face_tensor = face_tensor.to(self.device).unsqueeze(0)
            
            with torch.no_grad():
                embedding = self.facenet(face_tensor)
                embedding = F.normalize(embedding, p=2, dim=1)
            
            embedding_np = embedding.cpu().numpy().flatten()
            return embedding_np, True
            
        except Exception as e:
            logger.error(f"Fallback embedding error for {image_path}: {e}")
            return None, False
    
    def evaluate_images(self, positive_images: List[str], negative_images: List[str]) -> List[Dict]:
        """
        Evaluate all images and return results.
        
        Args:
            positive_images (List[str]): List of positive image paths
            negative_images (List[str]): List of negative image paths
            
        Returns:
            List[Dict]: Evaluation results
        """
        results = []
        all_images = [(img, 'positive') for img in positive_images] + [(img, 'negative') for img in negative_images]
        
        logger.info(f"Evaluating {len(all_images)} images...")
        
        for img_path, ground_truth in tqdm(all_images, desc="Evaluating images"):
            embedding, detected = self.get_face_embedding(img_path)
            
            if not detected or embedding is None:
                result = {
                    'filename': os.path.basename(img_path),
                    'similarity': 0.0,
                    'ground_truth': ground_truth,
                    'predicted': 'NOT MATCH',
                    'detected': False
                }
            else:
                # Compute similarity
                similarity = float(np.dot(embedding, self.person_a_mean))
                predicted = 'MATCH' if similarity >= self.threshold else 'NOT MATCH'
                
                result = {
                    'filename': os.path.basename(img_path),
                    'similarity': similarity,
                    'ground_truth': ground_truth,
                    'predicted': predicted,
                    'detected': True
                }
            
            results.append(result)
        
        return results
    
    def calculate_metrics(self, results: List[Dict]) -> Dict:
        """Calculate performance metrics from results."""
        tp = tn = fp = fn = 0
        detected_count = 0
        
        for result in results:
            if not result.get('detected', True):
                continue  # Skip undetected faces for metric calculation
            
            detected_count += 1
            ground_truth = result['ground_truth']
            predicted = result['predicted']
            
            if ground_truth == 'positive' and predicted == 'MATCH':
                tp += 1
            elif ground_truth == 'positive' and predicted == 'NOT MATCH':
                fn += 1
            elif ground_truth == 'negative' and predicted == 'MATCH':
                fp += 1
            elif ground_truth == 'negative' and predicted == 'NOT MATCH':
                tn += 1
        
        total = tp + tn + fp + fn
        accuracy = (tp + tn) / total if total > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        return {
            'total_images': len(results),
            'detected_images': detected_count,
            'undetected_images': len(results) - detected_count,
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
            'false_positive_rate': fpr,
            'threshold_used': self.threshold
        }
    
    def save_artifacts(
        self,
        run_folder: Path,
        results: List[Dict],
        metrics: Dict,
        metadata: Dict
    ):
        """Save all run artifacts."""
        
        # Save results.csv
        results_csv = run_folder / "results.csv"
        with open(results_csv, 'w', newline='') as f:
            if results:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
        
        # Save summary.json
        summary_json = run_folder / "summary.json"
        with open(summary_json, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save metadata.json
        metadata_json = run_folder / "metadata.json"
        with open(metadata_json, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved artifacts to {run_folder}")
    
    def create_visualizations(self, run_folder: Path, results: List[Dict]):
        """Create visualization plots."""
        try:
            # Extract similarities
            pos_similarities = [r['similarity'] for r in results if r['ground_truth'] == 'positive' and r.get('detected', True)]
            neg_similarities = [r['similarity'] for r in results if r['ground_truth'] == 'negative' and r.get('detected', True)]
            
            if not pos_similarities or not neg_similarities:
                logger.warning("Insufficient data for visualizations")
                return
            
            # Similarity histogram
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            plt.hist(pos_similarities, bins=30, alpha=0.7, label='Positive', color='green', density=True)
            plt.hist(neg_similarities, bins=30, alpha=0.7, label='Negative', color='red', density=True)
            plt.axvline(self.threshold, color='black', linestyle='--', label=f'Threshold ({self.threshold:.4f})')
            plt.xlabel('Similarity Score')
            plt.ylabel('Density')
            plt.title('Similarity Distribution')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # ROC-like visualization
            plt.subplot(1, 2, 2)
            thresholds = np.linspace(min(neg_similarities + pos_similarities), max(neg_similarities + pos_similarities), 100)
            tpr_list = []
            fpr_list = []
            
            for t in thresholds:
                tp = sum(1 for s in pos_similarities if s >= t)
                fn = sum(1 for s in pos_similarities if s < t)
                fp = sum(1 for s in neg_similarities if s >= t)
                tn = sum(1 for s in neg_similarities if s < t)
                
                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                
                tpr_list.append(tpr)
                fpr_list.append(fpr)
            
            plt.plot(fpr_list, tpr_list, 'b-', linewidth=2)
            plt.plot([0, 1], [0, 1], 'r--', alpha=0.5)
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(run_folder / "evaluation_plots.png", dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info("Created visualization plots")
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
    
    def print_summary(self, metrics: Dict, run_folder: Path):
        """Print evaluation summary."""
        print("\n" + "="*70)
        print("🎯 EXTENDED FACE VERIFICATION EVALUATION")
        print("="*70)
        
        print(f"📁 Run Folder: {run_folder}")
        print(f"📊 Total Images: {metrics['total_images']}")
        print(f"🔍 Detected Faces: {metrics['detected_images']}")
        print(f"❌ Undetected Faces: {metrics['undetected_images']}")
        
        cm = metrics['confusion_matrix']
        print(f"\n📈 Performance Metrics:")
        print(f"   Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"   Precision: {metrics['precision']:.4f}")
        print(f"   Recall: {metrics['recall']:.4f}")
        print(f"   F1-Score: {metrics['f1_score']:.4f}")
        print(f"   False Positive Rate: {metrics['false_positive_rate']:.4f} ({metrics['false_positive_rate']*100:.2f}%)")
        
        print(f"\n📋 Confusion Matrix:")
        print(f"   TP: {cm['true_positive']:3d} | TN: {cm['true_negative']:3d}")
        print(f"   FP: {cm['false_positive']:3d} | FN: {cm['false_negative']:3d}")
        
        print(f"\n🎯 Threshold: {metrics['threshold_used']:.4f}")
        
        # Warning for high FPR
        if metrics['false_positive_rate'] > 0.01:  # 1%
            print(f"\n⚠️  WARNING: False Positive Rate ({metrics['false_positive_rate']*100:.2f}%) exceeds 1%")
        else:
            print(f"\n✅ False Positive Rate within acceptable range (<1%)")
        
        print("="*70)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Extended Face Verification Evaluation")
    
    parser.add_argument(
        '--emb_path',
        type=str,
        default='data/embeddings/personA_normalized.npz',
        help='Path to PersonA embeddings (default: data/embeddings/personA_normalized.npz)'
    )
    
    parser.add_argument(
        '--query_dir',
        type=str,
        default='data/processed/query_images',
        help='Query images directory with positive/negative subfolders (default: data/processed/query_images)'
    )
    
    parser.add_argument(
        '--others_dir',
        type=str,
        default='data/processed/others',
        help='Directory containing negative sample pool (default: data/processed/others)'
    )
    
    parser.add_argument(
        '--num_neg',
        type=int,
        default=500,
        help='Number of negative samples to include (default: 500)'
    )
    
    parser.add_argument(
        '--add_pos_dir',
        type=str,
        default=None,
        help='Optional directory with additional positive images to include'
    )
    
    parser.add_argument(
        '--out_root',
        type=str,
        default='data/evaluation/runs',
        help='Root directory for evaluation runs (default: data/evaluation/runs)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducible sampling (default: 42)'
    )
    
    parser.add_argument(
        '--threshold_config',
        type=str,
        default='config/threshold.json',
        help='Path to threshold configuration (default: config/threshold.json)'
    )
    
    parser.add_argument(
        '--max_fpr',
        type=float,
        default=0.01,
        help='Maximum acceptable false positive rate (default: 0.01)'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize evaluator
        logger.info("Initializing Extended Evaluator...")
        evaluator = ExtendedEvaluator(
            emb_path=args.emb_path,
            threshold_config=args.threshold_config,
            seed=args.seed
        )
        
        # Sample negatives
        logger.info("Sampling negative images...")
        sampled_negatives = evaluator.sample_negatives(args.others_dir, args.num_neg)
        
        # Create run folder
        run_folder = evaluator.create_run_folder(args.out_root)
        
        # Setup run images
        logger.info("Setting up run images...")
        positive_images, negative_images = evaluator.setup_run_images(
            run_folder, args.query_dir, sampled_negatives, args.add_pos_dir
        )
        
        # Evaluate images
        results = evaluator.evaluate_images(positive_images, negative_images)
        
        # Calculate metrics
        metrics = evaluator.calculate_metrics(results)
        
        # Create metadata
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "emb_path": args.emb_path,
            "threshold_used": evaluator.threshold,
            "seed": args.seed,
            "num_neg_requested": args.num_neg,
            "num_neg_actual": len(sampled_negatives),
            "num_pos": len(positive_images),
            "run_id": run_folder.name,
            "model_version": "FaceNet vggface2",
            "additional_pos_dir": args.add_pos_dir
        }
        
        # Save artifacts
        evaluator.save_artifacts(run_folder, results, metrics, metadata)
        
        # Create visualizations
        evaluator.create_visualizations(run_folder, results)
        
        # Print summary
        evaluator.print_summary(metrics, run_folder)
        
        # Exit with error if FPR exceeds threshold
        if metrics['false_positive_rate'] > args.max_fpr:
            logger.error(f"FPR ({metrics['false_positive_rate']:.4f}) exceeds maximum ({args.max_fpr})")
            sys.exit(1)
        
        logger.info("Extended evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during extended evaluation: {str(e)}")
        raise


if __name__ == "__main__":
    main()