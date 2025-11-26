"""
Image preprocessing module for face verification project.

This module provides functions for loading, processing, and saving images
for face verification tasks. All functions are designed to work with both
Mac M2 and Google Colab environments.
"""

import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Union
import logging
import argparse
from pathlib import Path
import torch
from facenet_pytorch import MTCNN

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_image(path: str) -> np.ndarray:
    """
    Load an image from the specified path.
    
    Args:
        path (str): Path to the image file
        
    Returns:
        np.ndarray: Loaded image in RGB format
        
    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If image cannot be loaded
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found at path: {path}")
    
    try:
        # Load image using OpenCV (BGR format)
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            raise ValueError(f"Cannot load image from: {path}")
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        logger.info(f"Successfully loaded image: {path} with shape {img_rgb.shape}")
        return img_rgb
    
    except Exception as e:
        logger.error(f"Error loading image {path}: {str(e)}")
        raise


def resize_image(image: np.ndarray, size: Tuple[int, int] = (160, 160)) -> np.ndarray:
    """
    Resize image to specified dimensions.
    
    Args:
        image (np.ndarray): Input image in RGB format
        size (Tuple[int, int]): Target size (width, height). Default: (160, 160)
        
    Returns:
        np.ndarray: Resized image
        
    Raises:
        ValueError: If image is invalid or size is invalid
    """
    if image is None or len(image.shape) != 3:
        raise ValueError("Invalid image: must be a 3D RGB image")
    
    if len(size) != 2 or size[0] <= 0 or size[1] <= 0:
        raise ValueError("Size must be a tuple of two positive integers")
    
    try:
        # Use high-quality interpolation for resizing
        resized = cv2.resize(image, size, interpolation=cv2.INTER_CUBIC)
        
        logger.info(f"Resized image from {image.shape[:2]} to {size}")
        return resized
    
    except Exception as e:
        logger.error(f"Error resizing image: {str(e)}")
        raise


def normalize_image(image: np.ndarray, method: str = 'standard') -> np.ndarray:
    """
    Normalize image pixel values.
    
    Args:
        image (np.ndarray): Input image
        method (str): Normalization method. Options: 'standard', 'minmax', 'facenet'
                     - 'standard': (pixel - 127.5) / 128.0
                     - 'minmax': (pixel - min) / (max - min)
                     - 'facenet': Specific to FaceNet requirements
        
    Returns:
        np.ndarray: Normalized image
        
    Raises:
        ValueError: If invalid method or image
    """
    if image is None or len(image.shape) != 3:
        raise ValueError("Invalid image: must be a 3D RGB image")
    
    # Convert to float32 for processing
    img_float = image.astype(np.float32)
    
    try:
        if method == 'standard':
            # Standard normalization: values between -1 and 1
            normalized = (img_float - 127.5) / 128.0
            
        elif method == 'minmax':
            # Min-max normalization: values between 0 and 1
            min_val = np.min(img_float)
            max_val = np.max(img_float)
            normalized = (img_float - min_val) / (max_val - min_val)
            
        elif method == 'facenet':
            # FaceNet specific normalization
            normalized = (img_float - 127.5) / 128.0
            
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        logger.info(f"Normalized image using '{method}' method")
        return normalized
    
    except Exception as e:
        logger.error(f"Error normalizing image: {str(e)}")
        raise


def save_processed(image: np.ndarray, path: str, denormalize: bool = True) -> None:
    """
    Save processed image to specified path.
    
    Args:
        image (np.ndarray): Processed image to save
        path (str): Output path for the saved image
        denormalize (bool): Whether to denormalize the image before saving
        
    Raises:
        ValueError: If image is invalid
        IOError: If cannot save to specified path
    """
    if image is None or len(image.shape) != 3:
        raise ValueError("Invalid image: must be a 3D RGB image")
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Prepare image for saving
        if denormalize:
            # Assume image was normalized with 'standard' method
            if np.min(image) < 0:  # Likely normalized between -1 and 1
                img_to_save = ((image * 128.0) + 127.5).astype(np.uint8)
            else:  # Likely normalized between 0 and 1
                img_to_save = (image * 255).astype(np.uint8)
        else:
            img_to_save = image.astype(np.uint8)
        
        # Convert RGB to BGR for OpenCV saving
        img_bgr = cv2.cvtColor(img_to_save, cv2.COLOR_RGB2BGR)
        
        # Save the image
        success = cv2.imwrite(path, img_bgr)
        if not success:
            raise IOError(f"Failed to save image to: {path}")
        
        logger.info(f"Successfully saved processed image to: {path}")
    
    except Exception as e:
        logger.error(f"Error saving image to {path}: {str(e)}")
        raise


def preprocess_image(
    image_path: str,
    size: Tuple[int, int] = (160, 160),
    normalize_method: str = 'facenet'
) -> np.ndarray:
    """
    Complete preprocessing pipeline for a single image.
    
    Args:
        image_path (str): Path to input image
        size (Tuple[int, int]): Target size for resizing
        normalize_method (str): Normalization method to use
        
    Returns:
        np.ndarray: Fully preprocessed image ready for model input
    """
    try:
        # Load image
        image = load_image(image_path)
        
        # Resize image
        image = resize_image(image, size)
        
        # Normalize image
        image = normalize_image(image, normalize_method)
        
        logger.info(f"Successfully preprocessed image: {image_path}")
        return image
    
    except Exception as e:
        logger.error(f"Error in preprocessing pipeline for {image_path}: {str(e)}")
        raise


def batch_preprocess_images(
    input_dir: str,
    output_dir: str,
    size: Tuple[int, int] = (160, 160),
    normalize_method: str = 'facenet'
) -> int:
    """
    Preprocess all images in a directory.
    
    Args:
        input_dir (str): Directory containing input images
        output_dir (str): Directory to save processed images
        size (Tuple[int, int]): Target size for resizing
        normalize_method (str): Normalization method to use
        
    Returns:
        int: Number of images processed successfully
        
    Raises:
        FileNotFoundError: If input directory doesn't exist
    """
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Supported image extensions
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    processed_count = 0
    failed_count = 0
    
    for filename in os.listdir(input_dir):
        if any(filename.lower().endswith(ext) for ext in valid_extensions):
            try:
                input_path = os.path.join(input_dir, filename)
                output_path = os.path.join(output_dir, filename)
                
                # Preprocess image
                processed_img = preprocess_image(input_path, size, normalize_method)
                
                # Save processed image
                save_processed(processed_img, output_path)
                
                processed_count += 1
                
            except Exception as e:
                logger.error(f"Failed to process {filename}: {str(e)}")
                failed_count += 1
    
    logger.info(f"Batch processing complete: {processed_count} successful, {failed_count} failed")
    return processed_count


def visualize_preprocessing_steps(image_path: str, size: Tuple[int, int] = (160, 160)) -> None:
    """
    Visualize the preprocessing steps for debugging purposes.
    
    Args:
        image_path (str): Path to input image
        size (Tuple[int, int]): Target size for resizing
    """
    try:
        # Load original image
        original = load_image(image_path)
        
        # Resize image
        resized = resize_image(original, size)
        
        # Normalize image
        normalized = normalize_image(resized, 'facenet')
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(original)
        axes[0].set_title(f'Original\nShape: {original.shape}')
        axes[0].axis('off')
        
        # Resized image
        axes[1].imshow(resized)
        axes[1].set_title(f'Resized\nShape: {resized.shape}')
        axes[1].axis('off')
        
        # Normalized image (denormalized for display)
        display_normalized = ((normalized * 128.0) + 127.5).astype(np.uint8)
        axes[2].imshow(display_normalized)
        axes[2].set_title(f'Normalized\nRange: [{normalized.min():.2f}, {normalized.max():.2f}]')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        logger.error(f"Error visualizing preprocessing for {image_path}: {str(e)}")
        raise


def crop_and_align_face(image_path: str, output_size: int = 160, margin: float = 0.2) -> Optional[np.ndarray]:
    """
    Detect, crop and align face from image using MTCNN.
    
    Args:
        image_path (str): Path to input image
        output_size (int): Output face size (default 160 for FaceNet)
        margin (float): Margin around detected face
        
    Returns:
        Optional[np.ndarray]: Cropped and aligned face or None if no face detected
    """
    try:
        # Initialize MTCNN detector - use CPU to avoid MPS issues with MTCNN
        # MPS has issues with adaptive pooling operations in MTCNN
        device = 'cpu'  # Force CPU for MTCNN to avoid MPS adaptive pool errors
        mtcnn = MTCNN(
            image_size=output_size,
            margin=int(margin * output_size),
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=True,
            device=device
        )
        
        # Load image
        img = Image.open(image_path).convert('RGB')
        
        # Detect and crop face
        face_tensor = mtcnn(img)
        
        if face_tensor is not None:
            # Convert tensor back to numpy array (0-255 range)
            face_array = face_tensor.permute(1, 2, 0).numpy()
            face_array = ((face_array + 1) * 127.5).astype(np.uint8)
            
            logger.info(f"Successfully detected and cropped face from: {image_path}")
            return face_array
        else:
            logger.warning(f"No face detected in: {image_path}")
            return None
            
    except Exception as e:
        logger.error(f"Error processing {image_path}: {e}")
        return None


def process_directory(input_dir: str, output_dir: str, size: int = 160, detector: str = 'mtcnn'):
    """
    Process all images in input directory and save cropped faces to output directory.
    
    Args:
        input_dir (str): Input directory path
        output_dir (str): Output directory path
        size (int): Output face size
        detector (str): Face detector to use (currently only 'mtcnn')
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return
        
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    # Process each subdirectory in input
    processed_count = 0
    failed_count = 0
    
    for subdir in input_path.iterdir():
        if subdir.is_dir():
            subdir_name = subdir.name
            output_subdir = output_path / subdir_name
            output_subdir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Processing directory: {subdir_name}")
            
            # Process all images in subdirectory
            for img_file in subdir.iterdir():
                if img_file.suffix.lower() in image_extensions:
                    # Crop and align face
                    face_array = crop_and_align_face(str(img_file), output_size=size)
                    
                    if face_array is not None:
                        # Save cropped face
                        output_filename = output_subdir / f"{img_file.stem}_cropped{img_file.suffix}"
                        cv2.imwrite(str(output_filename), cv2.cvtColor(face_array, cv2.COLOR_RGB2BGR))
                        processed_count += 1
                        
                        if processed_count % 50 == 0:
                            logger.info(f"Processed {processed_count} images...")
                    else:
                        failed_count += 1
    
    logger.info(f"Processing complete!")
    logger.info(f"Successfully processed: {processed_count} images")
    logger.info(f"Failed to process: {failed_count} images")


def main():
    """
    Command line interface for face preprocessing.
    """
    parser = argparse.ArgumentParser(description='Face Detection and Preprocessing')
    parser.add_argument('--input_dir', type=str, required=True,
                      help='Input directory containing images')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Output directory for processed faces')
    parser.add_argument('--size', type=int, default=160,
                      help='Output face size (default: 160)')
    parser.add_argument('--detector', type=str, default='mtcnn', choices=['mtcnn'],
                      help='Face detector to use (default: mtcnn)')
    
    args = parser.parse_args()
    
    logger.info(f"Starting face preprocessing...")
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Face size: {args.size}x{args.size}")
    logger.info(f"Detector: {args.detector}")
    
    # Process directory
    process_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        size=args.size,
        detector=args.detector
    )


# Example usage and testing functions
if __name__ == "__main__":
    main()