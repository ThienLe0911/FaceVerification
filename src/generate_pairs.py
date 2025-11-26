#!/usr/bin/env python3
"""
generate_pairs.py - Generate positive and negative similarity pairs for threshold tuning

Purpose:
- Load PersonA embeddings (per_image and mean) from enrollment
- Generate positive similarities: PersonA per_image vs PersonA mean
- Generate negative similarities: Others images vs PersonA mean
- Save arrays for ROC/threshold analysis

Timeline Reference: github/timeline.txt - DAY 6 — Build Dataset for Threshold Tuning

Usage:
    python src/generate_pairs.py --emb_path data/embeddings/personA_normalized.npz \
                                 --others_dir data/processed/others \
                                 --out_dir data/evaluation \
                                 --num_neg 1000 \
                                 --seed 42

Output:
    data/evaluation/similarities_pos.npy  - Positive pair similarities
    data/evaluation/similarities_neg.npy  - Negative pair similarities  
    data/evaluation/neg_filenames.npy     - Negative filenames used
    data/evaluation/metadata.json         - Generation metadata

Dependencies:
    - numpy, pathlib
    - facenet-pytorch (fallback if inference.py not available)
    - torch, PIL (for fallback)
    - tqdm (optional, for progress bars)
"""

import os
import sys
import argparse
import json
import random
import logging
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple, Optional, Callable

import numpy as np

# Optional imports
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, **kwargs):
        return iterable

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    """L2 normalize embeddings to unit vectors."""
    norms = np.linalg.norm(embeddings, axis=-1, keepdims=True)
    norms = np.maximum(norms, 1e-12)  # Avoid division by zero
    return embeddings / norms


def check_normalization(embeddings: np.ndarray, name: str = "embeddings", atol: float = 1e-6) -> bool:
    """Check if embeddings are L2 normalized."""
    if embeddings.ndim == 1:
        norm = np.linalg.norm(embeddings)
        is_normalized = np.isclose(norm, 1.0, atol=atol)
        logger.info(f"{name}: norm = {norm:.6f}, normalized = {is_normalized}")
        return is_normalized
    else:
        norms = np.linalg.norm(embeddings, axis=-1)
        is_normalized = np.allclose(norms, 1.0, atol=atol)
        logger.info(f"{name}: norms range [{norms.min():.6f}, {norms.max():.6f}], normalized = {is_normalized}")
        return is_normalized


def get_embedding_function() -> Tuple[Callable[[str], np.ndarray], bool, str]:
    """
    Try to get embedding function in priority order:
    1. get_embedding_from_path from inference.py
    2. get_embedding_from_pil from inference.py (wrap with path loader)
    3. Fallback: facenet-pytorch setup
    
    Returns:
        (embedding_function, fallback_used, device_used)
    """
    fallback_used = False
    device_used = "unknown"
    
    # Try method 1: get_embedding_from_path
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from inference import get_embedding_from_path
        logger.info("✅ Using get_embedding_from_path from inference.py")
        return get_embedding_from_path, False, "inference_device"
    except (ImportError, AttributeError) as e:
        logger.debug(f"get_embedding_from_path not available: {e}")
    
    # Try method 2: get_embedding_from_pil (wrap with PIL loader)
    try:
        from inference import get_embedding_from_pil
        from PIL import Image
        
        def wrapped_embedding_from_path(image_path: str) -> np.ndarray:
            """Wrapper for get_embedding_from_pil that loads from path."""
            img = Image.open(image_path).convert('RGB')
            embedding = get_embedding_from_pil(img)
            if isinstance(embedding, (list, tuple)):
                embedding = np.array(embedding)
            return embedding.flatten() if embedding.ndim > 1 else embedding
            
        logger.info("✅ Using get_embedding_from_pil from inference.py (wrapped)")
        return wrapped_embedding_from_path, False, "inference_device"
    except (ImportError, AttributeError) as e:
        logger.debug(f"get_embedding_from_pil not available: {e}")
    
    # Try method 3: Fallback setup
    logger.warning("⚠️ inference.py methods not available, setting up fallback")
    return setup_fallback_embedding()


def setup_fallback_embedding() -> Tuple[Callable[[str], np.ndarray], bool, str]:
    """
    Setup fallback embedding model if inference.py is not available.
    Returns (mtcnn, resnet) or (None, None) if failed.
    """
    try:
        from facenet_pytorch import MTCNN, InceptionResnetV1
        import torch
        
        logger.info("Setting up fallback embedding models...")
        
        # Check for MPS availability
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device('mps')
            logger.info("Using MPS device for FaceNet")
        else:
            device = torch.device('cpu')
            logger.info("Using CPU device")
        
        # MTCNN for face detection (always CPU to avoid MPS issues)
        mtcnn = MTCNN(
            image_size=160,
            margin=0,
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=True,
            device='cpu'  # Force CPU for MTCNN
        )
        
        # FaceNet for embeddings
        resnet = InceptionResnetV1(pretrained='vggface2').eval()
        resnet = resnet.to(device)
        
        logger.info(f"Fallback models loaded successfully on {device}")
        return mtcnn, resnet, device
        
    except ImportError as e:
        logger.error(f"Failed to import facenet-pytorch: {e}")
        logger.error("Please install: pip install facenet-pytorch")
        return None, None, None

def get_embedding_fallback(image_path: str, mtcnn, resnet, device) -> Optional[np.ndarray]:
    """
    Fallback embedding function using facenet-pytorch directly.
    Returns normalized 512-dim embedding or None if failed.
    """
    try:
        # Load and detect face
        img = Image.open(image_path).convert('RGB')
        img_cropped = mtcnn(img)
        
        if img_cropped is None:
            logger.warning(f"No face detected in {image_path}")
            return None
        
        # Generate embedding
        img_cropped = img_cropped.unsqueeze(0).to(device)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            embedding = resnet(img_cropped).detach().cpu().numpy().flatten()
        
        # L2 normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
        
    except Exception as e:
        logger.warning(f"Failed to process {image_path}: {e}")
        return None

def load_embeddings(emb_path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load embeddings from npz file.
    Returns (per_image, mean, names)
    """
    if not os.path.exists(emb_path):
        logger.error(f"Embeddings file not found: {emb_path}")
        sys.exit(1)
    
    logger.info(f"Loading embeddings from {emb_path}")
    data = np.load(emb_path, allow_pickle=True)
    
    per_image = data['per_image']
    mean_emb = data['mean']
    names = data.get('names', [f"image_{i}" for i in range(len(per_image))])
    
    logger.info(f"Loaded {len(per_image)} images with {per_image.shape[1]}D embeddings")
    
    # Check normalization
    per_image_norms = np.linalg.norm(per_image, axis=1)
    mean_norm = np.linalg.norm(mean_emb)
    
    logger.info(f"Per-image norms: min={per_image_norms.min():.3f}, max={per_image_norms.max():.3f}")
    logger.info(f"Mean embedding norm: {mean_norm:.3f}")
    
    # Normalize if needed
    if not np.allclose(per_image_norms, 1.0, atol=1e-3):
        logger.warning("Per-image embeddings not normalized, normalizing now...")
        per_image = per_image / np.linalg.norm(per_image, axis=1, keepdims=True)
    
    if not np.isclose(mean_norm, 1.0, atol=1e-3):
        logger.warning("Mean embedding not normalized, normalizing now...")
        mean_emb = mean_emb / mean_norm
    
    return per_image, mean_emb, names

def get_negative_files(others_dir: str, num_neg: Optional[int] = None, seed: int = 42) -> List[str]:
    """
    Get list of negative image files from others directory.
    Optionally sample num_neg files randomly with given seed.
    """
    if not os.path.exists(others_dir):
        logger.error(f"Others directory not found: {others_dir}")
        return []
    
    # Find all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    all_files = []
    
    for root, dirs, files in os.walk(others_dir):
        for file in files:
            if Path(file).suffix.lower() in image_extensions:
                all_files.append(os.path.join(root, file))
    
    logger.info(f"Found {len(all_files)} image files in {others_dir}")
    
    # Sample if requested
    if num_neg is not None and num_neg < len(all_files):
        random.seed(seed)
        all_files = random.sample(all_files, num_neg)
        logger.info(f"Randomly sampled {len(all_files)} files (seed={seed})")
    
    return sorted(all_files)

def compute_positive_similarities(per_image: np.ndarray, mean_emb: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarities between each PersonA image and PersonA mean.
    Returns array of similarities.
    """
    # Cosine similarity = dot product (since both are normalized)
    similarities = np.dot(per_image, mean_emb)
    return similarities

def compute_negative_similarities(neg_files: List[str], mean_emb: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """
    Compute cosine similarities between negative images and PersonA mean.
    Returns (similarities, successful_filenames)
    """
    # Try to import from inference.py first
    embed_func = None
    try:
        sys.path.append('src')
        from inference import FaceVerifier
        from PIL import Image
        verifier = FaceVerifier()
        
        def path_to_embedding(path):
            """Wrapper to convert path to PIL image then get embedding."""
            with Image.open(path) as img:
                return verifier.get_embedding_from_pil(img.convert('RGB'))
        
        embed_func = path_to_embedding
        logger.info("Using FaceVerifier.get_embedding_from_pil from inference.py")
        
    except ImportError:
        logger.warning("Could not import from inference.py, using fallback...")
        mtcnn, resnet, device = setup_fallback_embedding()
        if mtcnn is None:
            logger.error("Failed to setup fallback embedding models")
            return np.array([]), []
        embed_func = lambda path: get_embedding_fallback(path, mtcnn, resnet, device)
    
    similarities = []
    successful_files = []
    
    logger.info(f"Processing {len(neg_files)} negative images...")
    
    for i, file_path in enumerate(neg_files):
        if i % 100 == 0 and i > 0:
            logger.info(f"Processed {i}/{len(neg_files)} negative images")
        
        try:
            embedding = embed_func(file_path)
            
            if embedding is not None:
                # Ensure normalization
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
                
                # Compute similarity
                similarity = np.dot(embedding, mean_emb)
                similarities.append(similarity)
                successful_files.append(os.path.basename(file_path))
            else:
                logger.warning(f"Failed to get embedding for {file_path} (returned None)")
        except Exception as e:
            logger.warning(f"Failed to get embedding for {file_path}: {e}")
    
    logger.info(f"Successfully processed {len(similarities)}/{len(neg_files)} negative images")
    return np.array(similarities), successful_files

def save_results(similarities_pos: np.ndarray, 
                similarities_neg: np.ndarray,
                neg_filenames: List[str],
                out_dir: str,
                metadata: dict):
    """
    Save similarities and metadata to output directory.
    """
    # Create output directory
    os.makedirs(out_dir, exist_ok=True)
    
    # Save arrays
    np.save(os.path.join(out_dir, 'similarities_pos.npy'), similarities_pos)
    np.save(os.path.join(out_dir, 'similarities_neg.npy'), similarities_neg)
    np.save(os.path.join(out_dir, 'neg_filenames.npy'), np.array(neg_filenames))
    
    # Save metadata
    with open(os.path.join(out_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Results saved to {out_dir}/")

def print_summary(similarities_pos: np.ndarray, similarities_neg: np.ndarray):
    """
    Print summary statistics of similarities.
    """
    logger.info("=" * 60)
    logger.info("SIMILARITY STATISTICS SUMMARY")
    logger.info("=" * 60)
    
    # Positive similarities
    logger.info(f"POSITIVE PAIRS (PersonA vs PersonA mean):")
    logger.info(f"  Count: {len(similarities_pos)}")
    logger.info(f"  Min:   {similarities_pos.min():.4f}")
    logger.info(f"  Max:   {similarities_pos.max():.4f}")
    logger.info(f"  Mean:  {similarities_pos.mean():.4f}")
    logger.info(f"  Std:   {similarities_pos.std():.4f}")
    
    # Negative similarities
    logger.info(f"NEGATIVE PAIRS (Others vs PersonA mean):")
    logger.info(f"  Count: {len(similarities_neg)}")
    logger.info(f"  Min:   {similarities_neg.min():.4f}")
    logger.info(f"  Max:   {similarities_neg.max():.4f}")
    logger.info(f"  Mean:  {similarities_neg.mean():.4f}")
    logger.info(f"  Std:   {similarities_neg.std():.4f}")
    
    # Separation analysis
    pos_min = similarities_pos.min()
    neg_max = similarities_neg.max()
    separation = pos_min - neg_max
    
    logger.info(f"SEPARATION ANALYSIS:")
    logger.info(f"  Positive min: {pos_min:.4f}")
    logger.info(f"  Negative max: {neg_max:.4f}")
    logger.info(f"  Separation:   {separation:.4f}")
    
    if separation > 0:
        logger.info(f"  ✅ Good separation! Easy to find threshold")
    else:
        logger.info(f"  ⚠️  Overlap detected. Threshold tuning needed")
    
    logger.info("=" * 60)

def main():
    parser = argparse.ArgumentParser(description="Generate positive and negative similarity pairs")
    parser.add_argument('--emb_path', default='data/embeddings/personA_normalized.npz',
                       help='Path to PersonA embeddings file')
    parser.add_argument('--others_dir', default='data/processed/others',
                       help='Directory containing processed other images')
    parser.add_argument('--out_dir', default='data/evaluation',
                       help='Output directory for results')
    parser.add_argument('--num_neg', type=int, default=None,
                       help='Number of negative samples (None = use all)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducible sampling')
    
    args = parser.parse_args()
    
    logger.info("🎯 Starting similarity pair generation...")
    logger.info(f"Embeddings: {args.emb_path}")
    logger.info(f"Others dir: {args.others_dir}")
    logger.info(f"Output dir: {args.out_dir}")
    logger.info(f"Num negatives: {args.num_neg or 'ALL'}")
    logger.info(f"Random seed: {args.seed}")
    
    # Load PersonA embeddings
    per_image, mean_emb, names = load_embeddings(args.emb_path)
    
    # Compute positive similarities
    logger.info("Computing positive similarities...")
    similarities_pos = compute_positive_similarities(per_image, mean_emb)
    
    # Get negative files
    neg_files = get_negative_files(args.others_dir, args.num_neg, args.seed)
    if not neg_files:
        logger.error("No negative files found!")
        sys.exit(1)
    
    # Compute negative similarities
    logger.info("Computing negative similarities...")
    similarities_neg, neg_filenames = compute_negative_similarities(neg_files, mean_emb)
    
    if len(similarities_neg) == 0:
        logger.error("No negative similarities computed!")
        sys.exit(1)
    
    # Prepare metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'n_images_pos': len(similarities_pos),
        'n_images_neg': len(similarities_neg),
        'embedding_dim': len(mean_emb),
        'seed': args.seed,
        'emb_path': args.emb_path,
        'others_dir': args.others_dir,
        'requested_neg': args.num_neg,
        'actual_neg': len(similarities_neg)
    }
    
    # Save results
    save_results(similarities_pos, similarities_neg, neg_filenames, args.out_dir, metadata)
    
    # Print summary
    print_summary(similarities_pos, similarities_neg)
    
    logger.info("🎉 Similarity pair generation completed successfully!")
    logger.info(f"📁 Results saved to: {args.out_dir}/")
    logger.info(f"📊 Ready for threshold tuning (DAY 7)!")

if __name__ == "__main__":
    main()