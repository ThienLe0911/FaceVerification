#!/usr/bin/env python3
"""
Check normalization status of current embeddings
"""
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def check_normalization(npz_path):
    """Check if embeddings are normalized."""
    print(f"🔍 Checking normalization for: {npz_path}")
    print("-" * 50)
    
    # Load embeddings
    d = np.load(npz_path, allow_pickle=True)
    per_image = d['per_image']  # (N,D)
    mean = d['mean']
    
    print(f"📊 Embeddings shape: {per_image.shape}")
    print(f"📊 Mean embedding shape: {mean.shape}")
    
    # Check current norms
    per_image_norms = np.linalg.norm(per_image, axis=1)
    mean_norm_value = np.linalg.norm(mean)
    
    print(f"\n📏 Current norms:")
    print(f"   Per-image norms - Min: {np.min(per_image_norms):.6f}")
    print(f"   Per-image norms - Max: {np.max(per_image_norms):.6f}")
    print(f"   Per-image norms - Mean: {np.mean(per_image_norms):.6f}")
    print(f"   Mean embedding norm: {mean_norm_value:.6f}")
    
    # Check if normalized (should be close to 1.0)
    is_normalized_per = np.allclose(per_image_norms, 1.0, atol=1e-6)
    is_normalized_mean = np.allclose(mean_norm_value, 1.0, atol=1e-6)
    
    print(f"\n✅ Normalization status:")
    print(f"   Per-image embeddings normalized: {is_normalized_per}")
    print(f"   Mean embedding normalized: {is_normalized_mean}")
    
    if not is_normalized_per or not is_normalized_mean:
        print(f"\n⚠️  Embeddings are NOT normalized - need to normalize!")
        return False
    else:
        print(f"\n✅ Embeddings are already normalized!")
        return True

def normalize_and_save(input_npz, output_npz):
    """Normalize embeddings and save to new file."""
    print(f"\n🔄 Normalizing embeddings...")
    print(f"Input: {input_npz}")
    print(f"Output: {output_npz}")
    
    # Load original embeddings
    d = np.load(input_npz, allow_pickle=True)
    per_image = d['per_image']  # (N,D)
    mean = d['mean']
    names = d['names'] if 'names' in d else None
    
    # Normalize per-image embeddings
    per_norm = per_image / np.linalg.norm(per_image, axis=1, keepdims=True)
    
    # Normalize mean embedding
    mean_norm = mean / np.linalg.norm(mean)
    
    # Verify normalization
    per_norms_check = np.linalg.norm(per_norm, axis=1)
    mean_norm_check = np.linalg.norm(mean_norm)
    
    print(f"✅ After normalization:")
    print(f"   Per-image norms - Max: {np.max(per_norms_check):.6f}")
    print(f"   Per-image norms - Min: {np.min(per_norms_check):.6f}")
    print(f"   Mean norm: {mean_norm_check:.6f}")
    
    # Save normalized embeddings
    if names is not None:
        np.savez(output_npz, per_image=per_norm, mean=mean_norm, names=names)
    else:
        np.savez(output_npz, per_image=per_norm, mean=mean_norm)
    
    print(f"💾 Saved normalized embeddings to: {output_npz}")

def main():
    print("🧮 Embedding Normalization Checker")
    print("=" * 50)
    
    # Check both embedding files
    embedding_files = [
        "data/embeddings/personA_train.npz",
        "data/embeddings/personA.npz"
    ]
    
    for emb_file in embedding_files:
        if os.path.exists(emb_file):
            is_normalized = check_normalization(emb_file)
            
            if not is_normalized:
                # Create normalized version
                output_file = emb_file.replace('.npz', '_normalized.npz')
                normalize_and_save(emb_file, output_file)
                
                # Verify the normalized version
                print(f"\n🔍 Verifying normalized file:")
                check_normalization(output_file)
            print("\n" + "="*50)
        else:
            print(f"⚠️  File not found: {emb_file}")

if __name__ == "__main__":
    main()