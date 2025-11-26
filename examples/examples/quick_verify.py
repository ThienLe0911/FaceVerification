#!/usr/bin/env python3
# examples/quick_verify.py
"""
Quick verification script to test face embeddings against query images.
Tests similarity between query images and enrolled personA embeddings.
"""
import numpy as np
from pathlib import Path
from PIL import Image
import argparse
import json

# Try to import your embedding utility (adapt if your project differs)
try:
    import sys
    import os
    # Add src directory to path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(os.path.dirname(current_dir), 'src')
    sys.path.insert(0, src_dir)
    
    from inference import get_embedding_from_path, get_embedding_from_pil
    print("✅ Successfully imported embedding functions")
except Exception as e:
    print(f"⚠️  Warning: Could not import embedding functions: {e}")
    get_embedding_from_path = None
    get_embedding_from_pil = None

def l2_normalize(x, eps=1e-12):
    """L2 normalize embedding vector."""
    x = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(x)
    if norm < eps:
        return x
    return x / norm

def embed_from_path(p):
    """Get embedding from image path."""
    if get_embedding_from_path is not None:
        e = get_embedding_from_path(str(p))
    else:
        # fallback: open PIL and call get_embedding_from_pil if exists
        img = Image.open(p).convert('RGB')
        if get_embedding_from_pil is not None:
            e = get_embedding_from_pil(img)
        else:
            raise RuntimeError("No embedding function found: implement get_embedding_from_path or get_embedding_from_pil")
    return np.asarray(e, dtype=np.float32)

def main(query_dir="data/processed/query_images", gallery_npz="data/embeddings/personA.npz",
         threshold=0.65, use_mean=False):
    """
    Main verification function.
    
    Args:
        query_dir (str): Directory with query images to test
        gallery_npz (str): Path to enrolled embeddings .npz file
        threshold (float): Similarity threshold for match decision
        use_mean (bool): Use mean embedding vs individual comparisons
    """
    print(f"📋 Quick Face Verification Test")
    print(f"================================")
    print(f"Query dir: {query_dir}")
    print(f"Gallery: {gallery_npz}")
    print(f"Threshold: {threshold}")
    print(f"Use mean: {use_mean}")
    print()
    
    qdir = Path(query_dir)
    if not qdir.exists():
        raise SystemExit(f"No query dir: {qdir}")

    # Load gallery embeddings
    d = np.load(gallery_npz, allow_pickle=True)
    per_image = d['per_image']  # shape (N,D)
    mean = d['mean']
    
    print(f"📊 Gallery info: {per_image.shape[0]} images, {per_image.shape[1]}D embeddings")

    # normalize
    per_image_norm = per_image / np.linalg.norm(per_image, axis=1, keepdims=True)
    mean_norm = mean / np.linalg.norm(mean)
    
    print()
    print("🔍 Testing query images:")
    print("-" * 50)

    results = []
    for p in sorted(qdir.iterdir()):
        if not p.is_file(): 
            continue
        try:
            q_emb = embed_from_path(p)
            q_emb = l2_normalize(q_emb)
        except Exception as e:
            print(f"⚠️  Skipping {p.name}: {e}")
            continue

        if use_mean:
            score = float(np.dot(q_emb, mean_norm))
            max_score = score
        else:
            sims = per_image_norm.dot(q_emb)
            max_score = float(np.max(sims))
            best_idx = int(np.argmax(sims))

        match = max_score >= threshold
        match_emoji = "✅" if match else "❌"
        
        result_line = f"{match_emoji} {p.name} -> score={max_score:.4f} -> MATCH={match}"
        if not use_mean:
            result_line += f" (best: #{best_idx})"
        
        print(result_line)
        results.append({
            'filename': p.name,
            'score': max_score,
            'match': match,
            'best_idx': best_idx if not use_mean else None
        })
    
    print()
    print("📈 Summary:")
    print("-" * 50)
    total = len(results)
    matches = sum(1 for r in results if r['match'])
    print(f"Total tests: {total}")
    print(f"Matches: {matches}")
    print(f"Non-matches: {total - matches}")
    if total > 0:
        print(f"Match rate: {matches/total*100:.1f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Quick face verification test')
    parser.add_argument('--query_dir', type=str, default='data/processed/query_images',
                      help='Directory with query images')
    parser.add_argument('--gallery_npz', type=str, default='data/embeddings/personA.npz',
                      help='Path to gallery embeddings file')
    parser.add_argument('--threshold', type=float, default=0.65,
                      help='Similarity threshold for match')
    parser.add_argument('--use_mean', action='store_true',
                      help='Use mean embedding instead of best match')
    
    args = parser.parse_args()
    main(query_dir=args.query_dir, gallery_npz=args.gallery_npz, 
         threshold=args.threshold, use_mean=args.use_mean)