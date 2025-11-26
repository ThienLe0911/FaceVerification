#!/usr/bin/env python3
"""
Simple Face Detection using OpenCV Haar Cascades
Thay thế face_recognition library để tránh dlib dependency
"""

import cv2
import numpy as np
from pathlib import Path

class SimpleFaceDetector:
    def __init__(self):
        """Initialize OpenCV cascade classifier"""
        try:
            # Load Haar cascade for face detection
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            if self.face_cascade.empty():
                raise Exception("Could not load face cascade")
            print("✅ OpenCV Haar Cascade Face Detector initialized")
        except Exception as e:
            print(f"❌ Error initializing face detector: {e}")
            self.face_cascade = None
    
    def detect_faces(self, image_path: str):
        """
        Detect faces in image using OpenCV
        Returns: List of face dictionaries with bbox and confidence
        """
        if self.face_cascade is None:
            return []
        
        try:
            # Read image
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"❌ Could not load image: {image_path}")
                return []
            
            # Convert to grayscale for detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            height, width = image.shape[:2]
            
            print(f"🔍 Detecting faces in {Path(image_path).name} ({width}x{height})")
            
            # Detect faces with multiple scales
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,     # Image pyramid scale
                minNeighbors=5,      # Minimum neighbors required
                minSize=(30, 30),    # Minimum face size
                maxSize=(width//2, height//2),  # Maximum face size
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            print(f"   Raw detection: {len(faces)} faces found")
            
            # Convert to our format and filter
            face_list = []
            for i, (x, y, w, h) in enumerate(faces):
                # Calculate quality metrics
                face_area = w * h
                image_area = width * height
                area_ratio = face_area / image_area
                
                # Quality scoring based on size - Made less strict
                if area_ratio < 0.003:  # Changed from 0.01 to 0.003 (0.3%)
                    quality_score = 0.2
                elif area_ratio > 0.5:  # Too large (likely false positive)
                    quality_score = 0.2
                elif 0.02 < area_ratio < 0.3:  # Good size faces
                    quality_score = 0.8 + min(area_ratio * 2, 0.2)
                elif 0.005 < area_ratio <= 0.02:  # Small but valid faces
                    quality_score = 0.5 + area_ratio * 10  # 0.55 to 0.7
                else:
                    quality_score = 0.4  # Very small faces
                
                # Filter out very low quality detections - made less strict
                if quality_score < 0.35:  # Changed from 0.4 to 0.35
                    print(f"   Face {i+1}: Filtered out (quality={quality_score:.2f}, size={w}x{h})")
                    continue
                
                face_data = {
                    "face_id": len(face_list) + 1,
                    "bbox": [int(x), int(y), int(x + w), int(y + h)],  # [x1, y1, x2, y2]
                    "confidence": round(quality_score, 3),
                    "width": int(w),
                    "height": int(h),
                    "area_ratio": round(area_ratio, 4)
                }
                
                face_list.append(face_data)
                print(f"   Face {len(face_list)}: bbox=({x},{y},{x+w},{y+h}), quality={quality_score:.2f}")
            
            print(f"   ✅ Final result: {len(face_list)} valid faces")
            return face_list
            
        except Exception as e:
            print(f"❌ Error detecting faces: {e}")
            return []
    
    def generate_fake_embedding(self, face_bbox):
        """
        Generate fake embedding for demo purposes
        In real system this would extract face features
        """
        x1, y1, x2, y2 = face_bbox
        
        # Use bbox coordinates to create deterministic "embedding"
        fake_features = [
            (x1 + x2) / 1000.0,  # Center X normalized
            (y1 + y2) / 1000.0,  # Center Y normalized  
            (x2 - x1) / 100.0,   # Width normalized
            (y2 - y1) / 100.0,   # Height normalized
        ]
        
        # Extend to 128 dimensions (typical face embedding size)
        embedding = fake_features * 32  # 4 * 32 = 128
        return np.array(embedding[:128])
    
    def compare_embeddings(self, embedding1, embedding2):
        """
        Compare two embeddings and return similarity score
        """
        # Cosine similarity
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        
        # Convert to 0-1 range and add some randomness for demo
        similarity = (similarity + 1) / 2  # [-1,1] -> [0,1]
        similarity = max(0.0, min(1.0, similarity))
        
        # Add controlled randomness for demo realism
        import random
        noise = random.uniform(-0.15, 0.15)
        similarity = max(0.0, min(1.0, similarity + noise))
        
        return similarity

def test_simple_detector():
    """Test the simple face detector"""
    print("🧪 Testing Simple Face Detector")
    print("=" * 50)
    
    detector = SimpleFaceDetector()
    
    # Check command line arguments first
    import sys
    if len(sys.argv) > 1:
        test_image = sys.argv[1]
        print(f"📷 Testing with command line image: {test_image}")
    else:
        # Test with our default image
        test_image = "data/raw/query_images/single_face/IMG_9703.jpg"
        print(f"📷 Testing with default image: {test_image}")
    
    if Path(test_image).exists():
        faces = detector.detect_faces(test_image)
        
        print(f"\n📊 Detection Results:")
        print(f"   Image: {test_image}")
        print(f"   Faces found: {len(faces)}")
        
        for face in faces:
            print(f"   Face {face['face_id']}: {face['bbox']}, confidence={face['confidence']}")
            
            # Test fake embedding
            embedding = detector.generate_fake_embedding(face['bbox'])
            print(f"     Embedding shape: {embedding.shape}")
            print(f"     Embedding sample: {embedding[:5]}")
    else:
        print(f"❌ Test image not found: {test_image}")
    
    return detector

if __name__ == "__main__":
    test_simple_detector()