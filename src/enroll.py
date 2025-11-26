# src/enroll.py
"""
Enrollment script for generating face embeddings from processed images.

This script processes all images in a directory, generates embeddings using FaceNet,
and saves them for later use in face verification tasks.
"""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import json
import torch
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import your project's FaceVerifier or get_embedding helper
try:
    from inference import FaceVerifier, get_embedding_from_pil
    has_faceverifier = True
except Exception as e:
    logger.warning(f"Could not import FaceVerifier: {e}")
    has_faceverifier = False

def pil_load_resize(path, size=(160,160)):
    """Load and resize image using PIL."""
    img = Image.open(path).convert('RGB')
    return img.resize(size)

def main(gallery_dir='data/processed/personA', out_path='data/embeddings/personA.npz', 
         meta_path='data/embeddings/personA_meta.json', device=None):
    """
    Main enrollment function.
    
    Args:
        gallery_dir (str): Directory containing processed face images
        out_path (str): Output path for embeddings .npz file
        meta_path (str): Output path for metadata .json file
        device (str): Device to use for computation
    """
    gallery_dir = Path(gallery_dir)
    out_path = Path(out_path)
    meta_path = Path(meta_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # choose device
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    logger.info(f"[enroll] Using device: {device}")

    # initialize FaceVerifier or fallback to building model from inference module
    verifier = None
    if has_faceverifier:
        try:
            verifier = FaceVerifier(device=device)
            logger.info("[enroll] FaceVerifier loaded successfully.")
        except Exception as e:
            logger.error(f"[enroll] Could not init FaceVerifier: {e}")
            verifier = None

    embeddings = []
    filenames = []
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    paths = []
    for ext in image_extensions:
        paths.extend(list(gallery_dir.glob(f'*{ext}')))
        paths.extend(list(gallery_dir.glob(f'*{ext.upper()}')))
    
    paths = sorted(paths)
    
    if len(paths) == 0:
        raise SystemExit(f"No images found in {gallery_dir}")
    
    logger.info(f"[enroll] Found {len(paths)} images to process")

    for p in paths:
        try:
            img = pil_load_resize(p)
            
            # Use FaceVerifier method if available
            if verifier is not None:
                emb = verifier.get_embedding_from_pil(img)
            else:
                # fallback: try to import InceptionResnetV1 directly (facenet-pytorch)
                logger.warning("[enroll] Using fallback method - loading models directly")
                from facenet_pytorch import InceptionResnetV1, MTCNN
                mtcnn = MTCNN(keep_all=False, device=device)
                model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
                
                # detect face crop (if needed)
                face = mtcnn(img)
                if face is None:
                    logger.warning(f"[enroll] No face detected in {p}; skipping")
                    continue
                    
                with torch.no_grad():
                    emb_tensor = model(face.unsqueeze(0).to(device))
                    emb = emb_tensor.cpu().numpy()[0]
            
            emb = np.asarray(emb, dtype=np.float32)
            embeddings.append(emb)
            filenames.append(p.name)
            logger.info(f"[enroll] Processed {p.name} -> embedding shape {emb.shape}")
            
        except Exception as e:
            logger.error(f"[enroll] Error processing {p}: {e}")

    if len(embeddings) == 0:
        raise SystemExit("No embeddings were created. Check detection/alignment steps.")

    # Stack embeddings and compute statistics
    per_image = np.stack(embeddings, axis=0)   # shape (N, D)
    mean_emb = np.mean(per_image, axis=0).astype(np.float32)
    
    # Save embeddings
    np.savez(out_path, per_image=per_image, mean=mean_emb, names=np.array(filenames))
    
    # Save metadata
    meta = {
        "num_images": len(filenames),
        "embedding_dim": int(per_image.shape[1]),
        "device": device,
        "source_folder": str(gallery_dir),
        "out_path": str(out_path),
        "filenames": filenames
    }
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    
    logger.info(f"[enroll] Successfully processed {len(filenames)} images")
    logger.info(f"[enroll] Embedding shape: {per_image.shape}")
    logger.info(f"[enroll] Saved embeddings to {out_path}")
    logger.info(f"[enroll] Saved metadata to {meta_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate face embeddings from processed images')
    parser.add_argument('--gallery_dir', type=str, default='data/processed/personA',
                      help='Directory containing processed face images')
    parser.add_argument('--out_path', type=str, default='data/embeddings/personA.npz',
                      help='Output path for embeddings file')
    parser.add_argument('--meta_path', type=str, default='data/embeddings/personA_meta.json',
                      help='Output path for metadata file')
    parser.add_argument('--device', type=str, default=None,
                      help='Device to use (cuda/mps/cpu)')
    
    args = parser.parse_args()
    main(gallery_dir=args.gallery_dir, out_path=args.out_path, 
         meta_path=args.meta_path, device=args.device)