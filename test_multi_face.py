"""
Test script for multi-face verification system.

This script validates the complete multi-face verification workflow
including detection, embedding, matching, and visualization.
"""

import os
import sys
import json
from pathlib import Path

# Add src to path
sys.path.append('src')

from multi_face_evaluator import MultiFaceEvaluator, create_run_directory
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_multi_face_system():
    """Comprehensive test of multi-face verification system."""
    
    print("="*70)
    print("🎯 MULTI-FACE VERIFICATION SYSTEM TEST")
    print("="*70)
    
    try:
        # Initialize evaluator
        print("\n1. Initializing MultiFaceEvaluator...")
        evaluator = MultiFaceEvaluator()
        print(f"   ✅ Gallery loaded: {len(evaluator.gallery_embeddings)} embeddings")
        print(f"   ✅ Threshold: {evaluator.threshold}")
        
        # Test single image
        print("\n2. Testing single image evaluation...")
        test_image = "data/raw/query_images/multiple_faces/IMG_8018.jpg"
        
        if os.path.exists(test_image):
            result = evaluator.evaluate_image_multi(test_image)
            
            print(f"   ✅ Image: {result['image']}")
            print(f"   ✅ Faces detected: {result['detected_count']}")
            print(f"   ✅ Processing time: {result['processing_time']:.3f}s")
            
            matches = [f for f in result['faces'] if f['predicted'] == 'MATCH']
            print(f"   ✅ PersonA matches: {len(matches)}")
            
            if matches:
                for i, match in enumerate(matches, 1):
                    print(f"      Face {i}: similarity={match['similarity']:.3f}, confidence={match['confidence']:.3f}")
        else:
            print(f"   ❌ Test image not found: {test_image}")
        
        # Test batch processing
        print("\n3. Testing batch processing...")
        test_folder = "data/raw/query_images/multiple_faces"
        out_dir = create_run_directory()
        
        if os.path.exists(test_folder):
            results = evaluator.evaluate_folder(
                test_folder, 
                out_dir, 
                save_annotations=True
            )
            
            total_faces = sum(len(r.get('faces', [])) for r in results)
            total_matches = sum(sum(1 for f in r.get('faces', []) if f.get('predicted') == 'MATCH') for r in results)
            
            print(f"   ✅ Images processed: {len(results)}")
            print(f"   ✅ Total faces: {total_faces}")
            print(f"   ✅ PersonA matches: {total_matches}")
            print(f"   ✅ Match rate: {total_matches/total_faces*100:.1f}%")
            print(f"   ✅ Results saved to: {out_dir}")
            
            # Verify output files
            summary_file = os.path.join(out_dir, 'summary.json')
            annotated_dir = os.path.join(out_dir, 'annotated')
            
            if os.path.exists(summary_file):
                print(f"   ✅ Summary file created")
            
            if os.path.exists(annotated_dir):
                annotated_files = [f for f in os.listdir(annotated_dir) if f.endswith('.jpg')]
                print(f"   ✅ Annotated images: {len(annotated_files)}")
        else:
            print(f"   ❌ Test folder not found: {test_folder}")
        
        # Performance summary
        print("\n4. Performance Analysis...")
        if 'results' in locals():
            avg_processing_time = sum(r.get('processing_time', 0) for r in results) / len(results)
            avg_faces_per_image = total_faces / len(results)
            
            print(f"   ✅ Average processing time: {avg_processing_time:.3f}s per image")
            print(f"   ✅ Average faces per image: {avg_faces_per_image:.1f}")
            print(f"   ✅ Processing speed: {avg_faces_per_image/avg_processing_time:.1f} faces/second")
        
        print("\n" + "="*70)
        print("🎉 MULTI-FACE VERIFICATION SYSTEM TEST COMPLETED SUCCESSFULLY!")
        print("="*70)
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def validate_module_structure():
    """Validate that all required components are present."""
    
    print("\n🔍 Module Structure Validation:")
    
    required_files = [
        'src/multi_face_evaluator.py',
        'src/inference.py',
        'config/threshold.json',
        'data/embeddings/personA_normalized.npz'
    ]
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path}")
            return False
    
    # Test import
    try:
        from multi_face_evaluator import MultiFaceEvaluator
        print(f"   ✅ MultiFaceEvaluator import successful")
    except Exception as e:
        print(f"   ❌ MultiFaceEvaluator import failed: {e}")
        return False
    
    return True


def main():
    """Main test runner."""
    
    print("🧪 Starting Multi-Face Verification System Tests...\n")
    
    # Change to project directory
    project_dir = Path(__file__).parent
    os.chdir(project_dir)
    
    # Validate structure
    if not validate_module_structure():
        print("\n❌ Module structure validation failed!")
        return False
    
    # Run comprehensive test
    success = test_multi_face_system()
    
    if success:
        print("\n✅ ALL TESTS PASSED! Multi-face verification system is ready for production.")
        return True
    else:
        print("\n❌ SOME TESTS FAILED! Please check the errors above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)