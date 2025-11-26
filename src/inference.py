"""
Face verification inference module.

This module provides functions for loading pretrained FaceNet models,
generating face embeddings, and computing similarity scores for face verification.
Designed to work with both Mac M2 and Google Colab environments.
"""

import os
import json
import torch
import torch.nn.functional as F
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import cv2
from typing import Tuple, List, Optional, Union
import logging
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_threshold(config_path: str = "config/threshold.json") -> float:
    """
    Load verification threshold from configuration file.
    
    Args:
        config_path (str): Path to threshold configuration JSON file
        
    Returns:
        float: Verification threshold value
    """
    fallback_threshold = 0.65
    
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            threshold = config.get("personA_threshold", fallback_threshold)
            logger.info(f"[THRESHOLD] Loaded threshold: {threshold}")
            return threshold
        else:
            logger.warning(f"[THRESHOLD] Config file not found: {config_path}, using fallback: {fallback_threshold}")
            return fallback_threshold
    except Exception as e:
        logger.error(f"[THRESHOLD] Error loading config: {e}, using fallback: {fallback_threshold}")
        return fallback_threshold


class FaceVerifier:
    """
    Face verification system using pretrained FaceNet model.
    
    This class handles face detection, embedding generation, and similarity computation
    for face verification tasks.
    """
    
    def __init__(
        self,
        device: Optional[str] = None,
        detection_threshold: float = 0.6,
        verification_threshold: Optional[float] = None
    ):
        """
        Initialize the Face Verifier.
        
        Args:
            device (Optional[str]): Device to use ('cuda', 'cpu', 'mps'). Auto-detect if None.
            detection_threshold (float): Confidence threshold for face detection
            verification_threshold (Optional[float]): Similarity threshold for verification. 
                                                     If None, loads from config/threshold.json
        """
        self.device = self._get_device(device)
        self.detection_threshold = detection_threshold
        
        # Load verification threshold from config if not provided
        if verification_threshold is None:
            self.verification_threshold = load_threshold()
        else:
            self.verification_threshold = verification_threshold
            logger.info(f"[THRESHOLD] Using provided threshold: {verification_threshold}")
        
        # Initialize models
        self.mtcnn = None
        self.facenet = None
        self._load_models()
        
        logger.info(f"FaceVerifier initialized on device: {self.device}")
    
    def _get_device(self, device: Optional[str] = None) -> str:
        """
        Determine the best available device.
        
        Args:
            device (Optional[str]): Preferred device
            
        Returns:
            str: Device to use
        """
        if device is not None:
            return device
        
        if torch.cuda.is_available():
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'  # Apple Silicon GPU
        else:
            return 'cpu'
    
    def _load_models(self):
        """Load MTCNN and FaceNet models."""
        try:
            # For Apple Silicon MPS compatibility, use CPU for MTCNN to avoid adaptive pool errors
            # and keep FaceNet on the specified device
            mtcnn_device = 'cpu'  # Force CPU for MTCNN to avoid MPS issues
            
            # Load MTCNN for face detection (on CPU)
            self.mtcnn = MTCNN(
                image_size=160,
                margin=0,
                min_face_size=20,
                thresholds=[0.6, 0.7, 0.7],  # P-Net, R-Net, O-Net thresholds
                factor=0.709,
                post_process=True,
                device=mtcnn_device
            )
            
            # Load pretrained FaceNet model (can use MPS/CUDA/CPU)
            self.facenet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
            
            logger.info(f"Successfully loaded MTCNN (CPU) and FaceNet ({self.device}) models")
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise
    
    def detect_face(self, image: Union[np.ndarray, str]) -> Optional[torch.Tensor]:
        """
        Detect and extract face from image.
        
        Args:
            image (Union[np.ndarray, str]): Input image or path to image
            
        Returns:
            Optional[torch.Tensor]: Detected face tensor or None if no face found
        """
        try:
            # Load image if path is provided
            if isinstance(image, str):
                if not os.path.exists(image):
                    raise FileNotFoundError(f"Image not found: {image}")
                img = Image.open(image).convert('RGB')
            else:
                # Convert numpy array to PIL Image
                if len(image.shape) == 3:
                    img = Image.fromarray(image.astype('uint8'), 'RGB')
                else:
                    raise ValueError("Invalid image format")
            
            # Detect face
            face_tensor = self.mtcnn(img)
            
            if face_tensor is not None:
                logger.info("Successfully detected face")
                return face_tensor
            else:
                logger.warning("No face detected in image")
                return None
                
        except Exception as e:
            logger.error(f"Error detecting face: {str(e)}")
            raise
    
    def generate_embedding(self, face_tensor: torch.Tensor) -> np.ndarray:
        """
        Generate face embedding using FaceNet.
        
        Args:
            face_tensor (torch.Tensor): Face tensor from MTCNN
            
        Returns:
            np.ndarray: Face embedding vector
        """
        try:
            if face_tensor is None:
                raise ValueError("Face tensor is None")
            
            # Ensure tensor is on correct device
            face_tensor = face_tensor.to(self.device)
            
            # Add batch dimension if needed
            if len(face_tensor.shape) == 3:
                face_tensor = face_tensor.unsqueeze(0)
            
            # Generate embedding
            with torch.no_grad():
                embedding = self.facenet(face_tensor)
                # Normalize embedding
                embedding = F.normalize(embedding, p=2, dim=1)
            
            # Convert to numpy array
            embedding_np = embedding.cpu().numpy().flatten()
            
            logger.info(f"Generated embedding with shape: {embedding_np.shape}")
            return embedding_np
            
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise
    
    def get_embedding_from_pil(self, pil_image: Image.Image) -> np.ndarray:
        """
        Generate embedding directly from PIL Image.
        
        Args:
            pil_image (Image.Image): PIL Image object
            
        Returns:
            np.ndarray: Face embedding vector
        """
        try:
            # Use MTCNN to detect and extract face
            face_tensor = self.mtcnn(pil_image)
            
            if face_tensor is None:
                raise ValueError("No face detected in the image")
            
            # Generate embedding
            return self.generate_embedding(face_tensor)
            
        except Exception as e:
            logger.error(f"Error generating embedding from PIL: {str(e)}")
            raise
    
    def compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
        method: str = 'cosine'
    ) -> float:
        """
        Compute similarity between two face embeddings.
        
        Args:
            embedding1 (np.ndarray): First face embedding
            embedding2 (np.ndarray): Second face embedding
            method (str): Similarity method ('cosine', 'euclidean')
            
        Returns:
            float: Similarity score
        """
        try:
            if method == 'cosine':
                # Cosine similarity
                similarity = cosine_similarity(
                    embedding1.reshape(1, -1),
                    embedding2.reshape(1, -1)
                )[0, 0]
                
            elif method == 'euclidean':
                # Euclidean distance (convert to similarity)
                distance = np.linalg.norm(embedding1 - embedding2)
                similarity = 1 / (1 + distance)  # Convert distance to similarity
                
            else:
                raise ValueError(f"Unknown similarity method: {method}")
            
            logger.info(f"Computed {method} similarity: {similarity:.4f}")
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error computing similarity: {str(e)}")
            raise
    
    def verify_faces(
        self,
        image1: Union[np.ndarray, str],
        image2: Union[np.ndarray, str],
        return_similarity: bool = False
    ) -> Union[bool, Tuple[bool, float]]:
        """
        Verify if two images contain the same person.
        
        Args:
            image1 (Union[np.ndarray, str]): First image or path
            image2 (Union[np.ndarray, str]): Second image or path
            return_similarity (bool): Whether to return similarity score
            
        Returns:
            Union[bool, Tuple[bool, float]]: Verification result and optionally similarity
        """
        try:
            # Detect faces
            face1 = self.detect_face(image1)
            face2 = self.detect_face(image2)
            
            if face1 is None or face2 is None:
                logger.warning("Could not detect face in one or both images")
                if return_similarity:
                    return False, 0.0
                return False
            
            # Generate embeddings
            embedding1 = self.generate_embedding(face1)
            embedding2 = self.generate_embedding(face2)
            
            # Compute similarity
            similarity = self.compute_similarity(embedding1, embedding2)
            
            # Make verification decision
            is_same_person = similarity > self.verification_threshold
            
            logger.info(f"Verification result: {is_same_person} (similarity: {similarity:.4f})")
            
            if return_similarity:
                return is_same_person, similarity
            return is_same_person
            
        except Exception as e:
            logger.error(f"Error in face verification: {str(e)}")
            if return_similarity:
                return False, 0.0
            return False
    
    def batch_verify_against_reference(
        self,
        reference_image: Union[np.ndarray, str],
        test_images: List[Union[np.ndarray, str]]
    ) -> List[Tuple[bool, float]]:
        """
        Verify multiple images against a reference image.
        
        Args:
            reference_image (Union[np.ndarray, str]): Reference image
            test_images (List[Union[np.ndarray, str]]): List of test images
            
        Returns:
            List[Tuple[bool, float]]: List of (is_match, similarity) for each test image
        """
        try:
            # Generate reference embedding
            ref_face = self.detect_face(reference_image)
            if ref_face is None:
                raise ValueError("Could not detect face in reference image")
            
            ref_embedding = self.generate_embedding(ref_face)
            
            results = []
            for i, test_image in enumerate(test_images):
                try:
                    # Detect test face
                    test_face = self.detect_face(test_image)
                    if test_face is None:
                        logger.warning(f"No face detected in test image {i}")
                        results.append((False, 0.0))
                        continue
                    
                    # Generate test embedding
                    test_embedding = self.generate_embedding(test_face)
                    
                    # Compute similarity
                    similarity = self.compute_similarity(ref_embedding, test_embedding)
                    
                    # Make decision
                    is_match = similarity > self.verification_threshold
                    results.append((is_match, similarity))
                    
                    logger.info(f"Test image {i}: {'Match' if is_match else 'No match'} "
                              f"(similarity: {similarity:.4f})")
                    
                except Exception as e:
                    logger.error(f"Error processing test image {i}: {str(e)}")
                    results.append((False, 0.0))
            
            return results
            
        except Exception as e:
            logger.error(f"Error in batch verification: {str(e)}")
            return [(False, 0.0)] * len(test_images)
    
    def get_face_embedding(self, image: Union[np.ndarray, str]) -> Optional[np.ndarray]:
        """
        Get face embedding from image (convenience method).
        
        Args:
            image (Union[np.ndarray, str]): Input image
            
        Returns:
            Optional[np.ndarray]: Face embedding or None if no face detected
        """
        try:
            face_tensor = self.detect_face(image)
            if face_tensor is None:
                return None
            
            return self.generate_embedding(face_tensor)
            
        except Exception as e:
            logger.error(f"Error getting face embedding: {str(e)}")
            return None
    
    def set_verification_threshold(self, threshold: float):
        """
        Update verification threshold.
        
        Args:
            threshold (float): New threshold value (0.0 to 1.0)
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0")
        
        self.verification_threshold = threshold
        logger.info(f"Updated verification threshold to: {threshold}")


def load_face_verifier(
    device: Optional[str] = None,
    verification_threshold: float = 0.8
) -> FaceVerifier:
    """
    Factory function to create and return a FaceVerifier instance.
    
    Args:
        device (Optional[str]): Device to use
        verification_threshold (float): Verification threshold
        
    Returns:
        FaceVerifier: Initialized face verifier
    """
    return FaceVerifier(device=device, verification_threshold=verification_threshold)


def quick_verify(
    image1: Union[np.ndarray, str],
    image2: Union[np.ndarray, str],
    threshold: float = 0.8
) -> Tuple[bool, float]:
    """
    Quick face verification function.
    
    Args:
        image1 (Union[np.ndarray, str]): First image
        image2 (Union[np.ndarray, str]): Second image
        threshold (float): Verification threshold
        
    Returns:
        Tuple[bool, float]: (is_same_person, similarity_score)
    """
    verifier = FaceVerifier(verification_threshold=threshold)
    return verifier.verify_faces(image1, image2, return_similarity=True)


def get_embedding_from_pil(pil_image: Image.Image, device: Optional[str] = None) -> np.ndarray:
    """
    Helper function to get embedding from PIL Image.
    
    Args:
        pil_image (Image.Image): PIL Image object
        device (Optional[str]): Device to use
        
    Returns:
        np.ndarray: Face embedding vector
    """
    verifier = FaceVerifier(device=device)
    return verifier.get_embedding_from_pil(pil_image)

def get_embedding_from_path(path):
    img = Image.open(path).convert('RGB')
    return get_embedding_from_pil(img)

# Example usage and testing
if __name__ == "__main__":
    print("Face Verification Inference Module")
    print("=================================")
    
    try:
        # Initialize verifier
        verifier = FaceVerifier()
        print(f"Verifier initialized on device: {verifier.device}")
        
        # Example with dummy data
        print("\nExample usage:")
        print("1. Load images with verifier.detect_face(image_path)")
        print("2. Generate embeddings with verifier.generate_embedding(face_tensor)")
        print("3. Compute similarity with verifier.compute_similarity(emb1, emb2)")
        print("4. Or use verifier.verify_faces(img1, img2) for end-to-end verification")
        
        print(f"\nCurrent settings:")
        print(f"- Detection threshold: {verifier.detection_threshold}")
        print(f"- Verification threshold: {verifier.verification_threshold}")
        print(f"- Device: {verifier.device}")
        
    except Exception as e:
        print(f"Error initializing verifier: {e}")
    
    print("\nModule loaded successfully. Ready for face verification!")