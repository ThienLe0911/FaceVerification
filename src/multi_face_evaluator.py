"""
Multi-face verification evaluation module.

This module provides functionality for detecting and verifying multiple faces in a single image
against a PersonA gallery. It supports batch processing, visualization, and detailed reporting.
"""

import os
import json
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Union
import logging
import argparse
from datetime import datetime
from tqdm import tqdm
from pathlib import Path
import time

# Import from existing inference module
from inference import FaceVerifier, load_threshold

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiFaceEvaluator:
    """
    Multi-face detection and verification system.
    
    Detects all faces in an image, extracts embeddings, and compares against PersonA gallery.
    """
    
    def __init__(
        self,
        emb_path: str = 'data/embeddings/personA_normalized.npz',
        threshold_config: str = 'config/threshold.json',
        device: Optional[str] = None
    ):
        """
        Initialize multi-face evaluator.
        
        Args:
            emb_path (str): Path to PersonA gallery embeddings
            threshold_config (str): Path to threshold configuration
            device (Optional[str]): Device to use for inference
        """
        self.emb_path = emb_path
        self.threshold_config = threshold_config
        self.device = device
        
        # Load threshold
        self.threshold = load_threshold(threshold_config)
        
        # Initialize face verifier
        self.face_verifier = FaceVerifier(
            device=device,
            verification_threshold=self.threshold
        )
        
        # Load PersonA gallery
        self._load_gallery()
        
        logger.info(f"MultiFaceEvaluator initialized with threshold: {self.threshold}")
    
    def _load_gallery(self):
        """Load PersonA gallery embeddings and metadata."""
        try:
            if not os.path.exists(self.emb_path):
                raise FileNotFoundError(f"Gallery embeddings not found: {self.emb_path}")
            
            # Load embeddings
            data = np.load(self.emb_path, allow_pickle=True)
            
            # Handle different file formats
            if 'embeddings' in data:
                # Format: embeddings, filenames
                self.gallery_embeddings = data['embeddings']
                self.gallery_filenames = data['filenames']
                self.gallery_mean = np.mean(self.gallery_embeddings, axis=0)
                self.gallery_mean = self.gallery_mean / np.linalg.norm(self.gallery_mean)
            elif 'per_image' in data:
                # Format: per_image, mean, names
                self.gallery_embeddings = data['per_image']
                self.gallery_filenames = data['names']
                self.gallery_mean = data['mean']
                # Mean should already be normalized, but ensure it
                if np.linalg.norm(self.gallery_mean) != 0:
                    self.gallery_mean = self.gallery_mean / np.linalg.norm(self.gallery_mean)
            else:
                raise ValueError(f"Unknown embedding file format. Keys: {list(data.keys())}")
            
            logger.info(f"Loaded PersonA gallery: {len(self.gallery_embeddings)} embeddings")
            
        except Exception as e:
            logger.error(f"Error loading gallery: {e}")
            raise
    
    def detect_all_faces(self, image_path: str) -> List[Dict]:
        """
        Detect all faces in an image and return bounding boxes and embeddings.
        
        Args:
            image_path (str): Path to input image
            
        Returns:
            List[Dict]: List of face detections with bbox, embedding, and confidence
        """
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            # Load image
            img = Image.open(image_path).convert('RGB')
            
            # Detect faces with bounding boxes and probabilities
            faces, probs = self.face_verifier.mtcnn.detect(img)
            
            detections = []
            
            if faces is not None and probs is not None:
                for i, (bbox, prob) in enumerate(zip(faces, probs)):
                    if prob > self.face_verifier.detection_threshold:
                        try:
                            # Use MTCNN to extract and align the face
                            # Create a list of just this bbox for extraction
                            face_tensor = self.face_verifier.mtcnn.extract(img, [bbox], save_path=None)
                            
                            if face_tensor is not None and len(face_tensor) > 0:
                                # face_tensor is a list, get the first (and only) tensor
                                face_img = face_tensor[0]
                                
                                # Generate embedding
                                embedding = self.face_verifier.generate_embedding(face_img)
                                
                                detection = {
                                    'bbox': bbox.tolist(),  # [x1, y1, x2, y2]
                                    'confidence': float(prob),
                                    'embedding': embedding,
                                    'face_id': i
                                }
                                detections.append(detection)
                                
                        except Exception as e:
                            logger.warning(f"Failed to process face {i}: {e}")
                            
                            # Fallback: manual crop and resize
                            try:
                                x1, y1, x2, y2 = bbox
                                # Ensure coordinates are within image bounds
                                img_w, img_h = img.size
                                x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(img_w, int(x2)), min(img_h, int(y2))
                                
                                # Crop face
                                face_crop = img.crop((x1, y1, x2, y2))
                                
                                # Resize to 160x160 for FaceNet
                                face_resized = face_crop.resize((160, 160), Image.LANCZOS)
                                
                                # Convert to tensor
                                face_array = np.array(face_resized).astype(np.float32) / 255.0
                                face_tensor = torch.from_numpy(face_array).permute(2, 0, 1)  # CHW format
                                
                                # Normalize (FaceNet expects values in [-1, 1])
                                face_tensor = (face_tensor - 0.5) / 0.5
                                
                                # Generate embedding
                                embedding = self.face_verifier.generate_embedding(face_tensor)
                                
                                detection = {
                                    'bbox': bbox.tolist(),
                                    'confidence': float(prob),
                                    'embedding': embedding,
                                    'face_id': i
                                }
                                detections.append(detection)
                                
                            except Exception as e2:
                                logger.warning(f"Fallback processing also failed for face {i}: {e2}")
                                continue
            
            logger.info(f"Detected {len(detections)} faces in {os.path.basename(image_path)}")
            return detections
            
        except Exception as e:
            logger.error(f"Error detecting faces in {image_path}: {e}")
            raise
    
    def evaluate_face(self, face_embedding: np.ndarray) -> Dict:
        """
        Evaluate a single face embedding against PersonA gallery.
        
        Args:
            face_embedding (np.ndarray): Face embedding vector
            
        Returns:
            Dict: Evaluation results with similarities and predictions
        """
        try:
            # Compute similarity to gallery mean
            similarity_to_mean = float(np.dot(face_embedding, self.gallery_mean))
            
            # Compute similarities to individual gallery embeddings
            individual_similarities = []
            for i, gallery_emb in enumerate(self.gallery_embeddings):
                sim = float(np.dot(face_embedding, gallery_emb))
                individual_similarities.append({
                    'similarity': sim,
                    'filename': str(self.gallery_filenames[i])
                })
            
            # Find best match
            best_match = max(individual_similarities, key=lambda x: x['similarity'])
            max_similarity = best_match['similarity']
            best_match_filename = best_match['filename']
            
            # Make prediction based on mean similarity
            predicted = "MATCH" if similarity_to_mean >= self.threshold else "NOT MATCH"
            
            return {
                'similarity_to_mean': similarity_to_mean,
                'max_similarity': max_similarity,
                'best_match_filename': best_match_filename,
                'predicted': predicted,
                'threshold': self.threshold
            }
            
        except Exception as e:
            logger.error(f"Error evaluating face: {e}")
            raise
    
    def evaluate_image_multi(
        self,
        path_to_image: str,
        emb_path: Optional[str] = None,
        threshold_config: Optional[str] = None,
        return_boxes: bool = True
    ) -> Dict:
        """
        Evaluate multiple faces in a single image.
        
        Args:
            path_to_image (str): Path to input image
            emb_path (Optional[str]): Path to gallery embeddings (override default)
            threshold_config (Optional[str]): Path to threshold config (override default)
            return_boxes (bool): Whether to include bounding boxes in results
            
        Returns:
            Dict: Results containing all detected faces and their evaluations
        """
        start_time = time.time()
        
        try:
            # Update paths if provided
            if emb_path and emb_path != self.emb_path:
                self.emb_path = emb_path
                self._load_gallery()
            
            if threshold_config and threshold_config != self.threshold_config:
                self.threshold = load_threshold(threshold_config)
                self.face_verifier.verification_threshold = self.threshold
            
            # Detect all faces
            detections = self.detect_all_faces(path_to_image)
            
            # Evaluate each face
            faces_results = []
            for detection in detections:
                evaluation = self.evaluate_face(detection['embedding'])
                
                face_result = {
                    'similarity': evaluation['similarity_to_mean'],
                    'max_similarity': evaluation['max_similarity'],
                    'predicted': evaluation['predicted'],
                    'best_match_filename': evaluation['best_match_filename'],
                    'confidence': detection['confidence']
                }
                
                if return_boxes:
                    face_result['bbox'] = detection['bbox']
                
                faces_results.append(face_result)
            
            processing_time = time.time() - start_time
            
            # Prepare result
            result = {
                'image': os.path.basename(path_to_image),
                'faces': faces_results,
                'detected_count': len(faces_results),
                'timestamp': datetime.now().isoformat(),
                'processing_time': round(processing_time, 3),
                'threshold': self.threshold
            }
            
            logger.info(f"Processed {os.path.basename(path_to_image)}: {len(faces_results)} faces in {processing_time:.3f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error evaluating image {path_to_image}: {e}")
            # Return error result
            return {
                'image': os.path.basename(path_to_image),
                'faces': [],
                'detected_count': 0,
                'detected': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def draw_boxes(
        self,
        input_image_path: str,
        faces: List[Dict],
        out_path: str,
        show_details: bool = True
    ):
        """
        Draw bounding boxes and labels on the image.
        
        Args:
            input_image_path (str): Path to input image
            faces (List[Dict]): List of face results from evaluate_image_multi
            out_path (str): Path to save annotated image
            show_details (bool): Whether to show similarity scores
        """
        try:
            # Load image
            img = Image.open(input_image_path).convert('RGB')
            draw = ImageDraw.Draw(img)
            
            # Try to use a font (fallback to default if not available)
            try:
                font = ImageFont.truetype("arial.ttf", 12)
            except:
                font = ImageFont.load_default()
            
            # Draw each face
            for i, face in enumerate(faces):
                if 'bbox' not in face:
                    continue
                    
                bbox = face['bbox']
                x1, y1, x2, y2 = bbox
                
                # Choose color based on prediction
                color = (0, 255, 0) if face['predicted'] == "MATCH" else (255, 0, 0)
                
                # Draw bounding box
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                
                # Prepare label
                if show_details:
                    label = f"{face['predicted']}\n{face['similarity']:.3f}"
                    if 'confidence' in face:
                        label += f"\nConf: {face['confidence']:.3f}"
                else:
                    label = face['predicted']
                
                # Draw label background
                text_bbox = draw.textbbox((x1, y1-40), label, font=font)
                draw.rectangle([text_bbox[0]-2, text_bbox[1]-2, text_bbox[2]+2, text_bbox[3]+2], 
                             fill=color, outline=color)
                
                # Draw label text
                draw.text((x1, y1-40), label, fill=(255, 255, 255), font=font)
            
            # Create output directory if needed
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            
            # Save annotated image
            img.save(out_path)
            logger.info(f"Saved annotated image: {out_path}")
            
        except Exception as e:
            logger.error(f"Error drawing boxes: {e}")
            raise
    
    def evaluate_folder(
        self,
        folder_path: str,
        out_dir: str,
        recursive: bool = False,
        save_annotations: bool = True
    ) -> List[Dict]:
        """
        Process multiple images in a folder.
        
        Args:
            folder_path (str): Path to folder containing images
            out_dir (str): Output directory for results
            recursive (bool): Whether to search recursively
            save_annotations (bool): Whether to save annotated images
            
        Returns:
            List[Dict]: Results for all processed images
        """
        try:
            # Find image files
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
            
            if recursive:
                image_files = []
                for root, dirs, files in os.walk(folder_path):
                    for file in files:
                        if Path(file).suffix.lower() in image_extensions:
                            image_files.append(os.path.join(root, file))
            else:
                image_files = [
                    os.path.join(folder_path, f) 
                    for f in os.listdir(folder_path)
                    if Path(f).suffix.lower() in image_extensions
                ]
            
            if not image_files:
                logger.warning(f"No image files found in {folder_path}")
                return []
            
            # Create output directories
            os.makedirs(out_dir, exist_ok=True)
            if save_annotations:
                annotations_dir = os.path.join(out_dir, 'annotated')
                os.makedirs(annotations_dir, exist_ok=True)
            
            # Process each image
            results = []
            for image_path in tqdm(image_files, desc="Processing images"):
                try:
                    # Evaluate image
                    result = self.evaluate_image_multi(image_path)
                    results.append(result)
                    
                    # Save annotated image
                    if save_annotations and 'faces' in result and result['faces']:
                        base_name = Path(image_path).stem
                        out_path = os.path.join(annotations_dir, f"{base_name}_annotated.jpg")
                        self.draw_boxes(image_path, result['faces'], out_path)
                    
                except Exception as e:
                    logger.error(f"Error processing {image_path}: {e}")
                    continue
            
            # Save batch results
            results_file = os.path.join(out_dir, 'batch_results.json')
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            # Generate summary
            total_images = len(results)
            total_faces = sum(len(r.get('faces', [])) for r in results)
            matches = sum(sum(1 for f in r.get('faces', []) if f.get('predicted') == 'MATCH') for r in results)
            
            summary = {
                'total_images': total_images,
                'total_faces_detected': total_faces,
                'total_matches': matches,
                'match_rate': round(matches / total_faces * 100, 2) if total_faces > 0 else 0,
                'processing_timestamp': datetime.now().isoformat()
            }
            
            summary_file = os.path.join(out_dir, 'summary.json')
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"Batch processing completed: {total_images} images, {total_faces} faces, {matches} matches")
            return results
            
        except Exception as e:
            logger.error(f"Error processing folder {folder_path}: {e}")
            raise


def create_run_directory() -> str:
    """Create timestamped run directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"data/evaluation/runs/{timestamp}_multiface"
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, 'annotated'), exist_ok=True)
    return run_dir


def main():
    """CLI interface for multi-face evaluation."""
    parser = argparse.ArgumentParser(description='Multi-face verification evaluation')
    parser.add_argument('--image', type=str, help='Path to input image')
    parser.add_argument('--folder', type=str, help='Path to folder with images')
    parser.add_argument('--out_dir', type=str, help='Output directory')
    parser.add_argument('--emb_path', type=str, default='data/embeddings/personA_normalized.npz',
                       help='Path to PersonA embeddings')
    parser.add_argument('--threshold_config', type=str, default='config/threshold.json',
                       help='Path to threshold configuration')
    parser.add_argument('--recursive', action='store_true', help='Search folders recursively')
    parser.add_argument('--no_annotations', action='store_true', help='Skip saving annotated images')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.image and not args.folder:
        parser.error("Either --image or --folder must be specified")
    
    if args.image and args.folder:
        parser.error("Cannot specify both --image and --folder")
    
    # Create output directory
    if args.out_dir:
        out_dir = args.out_dir
    else:
        out_dir = create_run_directory()
    
    # Initialize evaluator
    evaluator = MultiFaceEvaluator(
        emb_path=args.emb_path,
        threshold_config=args.threshold_config
    )
    
    if args.image:
        # Single image processing
        result = evaluator.evaluate_image_multi(args.image)
        
        # Print JSON result
        print(json.dumps(result, indent=2))
        
        # Save annotated image if faces detected
        if not args.no_annotations and result.get('faces'):
            os.makedirs(os.path.join(out_dir, 'annotated'), exist_ok=True)
            base_name = Path(args.image).stem
            out_path = os.path.join(out_dir, 'annotated', f"{base_name}_annotated.jpg")
            evaluator.draw_boxes(args.image, result['faces'], out_path)
        
        # Save result
        result_file = os.path.join(out_dir, 'result.json')
        os.makedirs(out_dir, exist_ok=True)
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
    
    elif args.folder:
        # Folder processing
        results = evaluator.evaluate_folder(
            args.folder,
            out_dir,
            recursive=args.recursive,
            save_annotations=not args.no_annotations
        )
        
        # Print summary
        total_faces = sum(len(r.get('faces', [])) for r in results)
        matches = sum(sum(1 for f in r.get('faces', []) if f.get('predicted') == 'MATCH') for r in results)
        
        print(f"\nBatch Processing Summary:")
        print(f"Images processed: {len(results)}")
        print(f"Total faces detected: {total_faces}")
        print(f"PersonA matches found: {matches}")
        print(f"Results saved to: {out_dir}")


def self_test():
    """Self-test function to validate the module."""
    logger.info("Running self-test...")
    
    # Check for test images
    test_dir = "data/raw/query_images/multiple_faces"
    if not os.path.exists(test_dir):
        logger.warning(f"Test directory not found: {test_dir}")
        return
    
    # Find test images
    image_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        logger.warning(f"No test images found in {test_dir}")
        return
    
    # Test with first 3 images
    test_images = image_files[:3]
    logger.info(f"Testing with {len(test_images)} images: {test_images}")
    
    try:
        # Initialize evaluator
        evaluator = MultiFaceEvaluator()
        
        # Process each test image
        for image_file in test_images:
            image_path = os.path.join(test_dir, image_file)
            logger.info(f"\n--- Testing: {image_file} ---")
            
            result = evaluator.evaluate_image_multi(image_path)
            
            print(f"Image: {result['image']}")
            print(f"Faces detected: {result['detected_count']}")
            
            for i, face in enumerate(result['faces']):
                print(f"  Face {i+1}: {face['predicted']} (similarity: {face['similarity']:.3f})")
            
            print(f"Processing time: {result.get('processing_time', 0):.3f}s")
        
        logger.info("Self-test completed successfully!")
        
    except Exception as e:
        logger.error(f"Self-test failed: {e}")


if __name__ == "__main__":
    if len(os.sys.argv) == 1:
        # No arguments provided, run self-test
        self_test()
    else:
        # Run CLI
        main()