#!/usr/bin/env python3
"""
Check for data leakage - verify test images are not in training embeddings
"""
import numpy as np
import os

def check_data_leakage():
    """Check if test images appear in enrolled embeddings (data leakage check)."""
    print("🔍 Data Leakage Detection")
    print("=" * 50)
    
    # Load normalized embeddings
    embeddings_file = 'data/embeddings/personA_normalized.npz'
    
    if not os.path.exists(embeddings_file):
        print(f"❌ Embeddings file not found: {embeddings_file}")
        return
    
    d = np.load(embeddings_file, allow_pickle=True)
    enrolled_names = list(d['names'])
    
    print(f"📊 Enrolled embeddings: {len(enrolled_names)} images")
    print("📋 Enrolled names:")
    for i, name in enumerate(enrolled_names):
        print(f"   {i+1:2d}. {name}")
    
    print("\n" + "-" * 50)
    
    # Get all test images from query directory
    query_dir = 'data/processed/query_images'
    if os.path.exists(query_dir):
        test_images = [f for f in os.listdir(query_dir) 
                      if f.endswith(('.jpg', '.png', '.jpeg'))]
        test_images.sort()
    else:
        test_images = []
    
    print(f"🧪 Test images in query directory: {len(test_images)}")
    
    # Specific test images we're concerned about (positive cases)
    held_out = ["pos4.png", "test_pos1.jpg", "test_pos2.jpg", "test_pos3.jpg"]
    
    # Also add any pos1.png, pos2.png, pos3.png if they exist
    for i in [1, 2, 3]:
        pos_file = f"pos{i}.png"
        if pos_file in test_images:
            held_out.append(pos_file)
    
    print(f"🎯 Key test images to check: {len(held_out)}")
    print("\n🔍 Data Leakage Check Results:")
    print("-" * 30)
    
    leakage_found = False
    
    for test_img in held_out:
        is_leaked = test_img in enrolled_names
        status = "❌ LEAKED" if is_leaked else "✅ SAFE"
        print(f"   {test_img:<15} in enrolled? {is_leaked:<5} {status}")
        
        if is_leaked:
            leakage_found = True
    
    print("\n" + "=" * 50)
    
    if leakage_found:
        print("🚨 DATA LEAKAGE DETECTED!")
        print("   Some test images are in the training set.")
        print("   Results will be artificially inflated.")
        print("   ⚠️  Need to retrain with proper train/test split!")
    else:
        print("✅ NO DATA LEAKAGE FOUND!")
        print("   All test images are properly held out.")
        print("   Results are valid and reliable.")
    
    # Additional check: show all test images status
    print(f"\n📋 All test images status:")
    print("-" * 30)
    
    for test_img in sorted(test_images):
        is_leaked = test_img in enrolled_names
        status = "❌ LEAKED" if is_leaked else "✅ SAFE"
        img_type = "POS" if test_img.startswith(('pos', 'test_pos')) else "NEG"
        print(f"   {test_img:<20} [{img_type}] {status}")

if __name__ == "__main__":
    check_data_leakage()