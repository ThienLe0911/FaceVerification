#!/usr/bin/env python3
"""
Quick verification script to test face embeddings against query images.
Simple version with direct imports.
"""
import numpy as np
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from inference import get_embedding_from_path, get_embedding_from_pil
    print("✅ Successfully imported embedding functions")
except Exception as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
from PIL import Image

def l2_normalize(x, eps=1e-12):
    """L2 normalize embedding vector."""
    x = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(x)
    if norm < eps:
        return x
    return x / norm

def main():
    """Main verification function."""
    print("📋 Quick Face Verification Test")
    print("=" * 50)
    
    # Hardcoded paths for simplicity
    query_dir = "data/processed/query_images"
    gallery_npz = "data/embeddings/personA_train.npz"  # Use new training set (27 images)
    threshold = 0.65
    
    if not os.path.exists(query_dir):
        print(f"❌ Query directory not found: {query_dir}")
        return
    
    if not os.path.exists(gallery_npz):
        print(f"❌ Gallery embeddings not found: {gallery_npz}")
        return

    # Load gallery embeddings
    print(f"📂 Loading gallery: {gallery_npz}")
    d = np.load(gallery_npz, allow_pickle=True)
    per_image = d['per_image']  # shape (N,D)
    mean = d['mean']
    
    print(f"📊 Gallery: {per_image.shape[0]} images, {per_image.shape[1]}D embeddings")

    # Normalize gallery embeddings
    per_image_norm = per_image / np.linalg.norm(per_image, axis=1, keepdims=True)
    mean_norm = mean / np.linalg.norm(mean)
    
    print(f"🔍 Testing images in: {query_dir}")
    print("-" * 50)

    # Test each query image
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    results = []
    
    for filename in sorted(os.listdir(query_dir)):
        if not any(filename.lower().endswith(ext) for ext in image_extensions):
            continue
            
        filepath = os.path.join(query_dir, filename)
        
        try:
            # Get embedding for query image  
            q_emb = get_embedding_from_path(filepath)
            q_emb = l2_normalize(q_emb)
            
            # Compute similarities with all gallery images
            sims = per_image_norm.dot(q_emb)
            max_score = float(np.max(sims))
            best_idx = int(np.argmax(sims))
            
            # Also compute similarity with mean
            mean_score = float(np.dot(q_emb, mean_norm))
            
            # Decide match based on threshold
            match = max_score >= threshold
            match_emoji = "✅" if match else "❌"
            
            result = f"{match_emoji} {filename:15} -> max={max_score:.4f} mean={mean_score:.4f} -> MATCH={match}"
            print(result)
            
            results.append({
                'filename': filename,
                'max_score': max_score,
                'mean_score': mean_score,
                'match': match,
                'best_idx': best_idx
            })
            
        except Exception as e:
            print(f"⚠️  Error processing {filename}: {e}")
    
    # Summary
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
    
    print(f"Threshold used: {threshold}")

if __name__ == "__main__":
    main()