#!/usr/bin/env python3
"""
Compare MTCNN vs OpenCV Haar Cascade detection on IMG_9569.png
"""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import torch
from facenet_pytorch import MTCNN

def test_opencv_detection(image_path: str):
    """Test OpenCV Haar Cascade detection"""
    print("🔍 Testing OpenCV Haar Cascade Detection")
    
    # Load cascade
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Load image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = image.shape[:2]
    
    print(f"   Image size: {width}x{height}")
    
    # Detect faces
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        maxSize=(width//2, height//2),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    print(f"   Raw detections: {len(faces)}")
    
    for i, (x, y, w, h) in enumerate(faces):
        area_ratio = (w * h) / (width * height)
        print(f"     Face {i+1}: ({x},{y},{w},{h}), area_ratio={area_ratio:.6f}")
    
    return faces

def test_mtcnn_detection(image_path: str):
    """Test MTCNN detection"""
    print("\n🧠 Testing MTCNN Detection")
    
    # Initialize MTCNN
    device = 'cpu'
    mtcnn = MTCNN(
        image_size=160,
        margin=32,  # 0.2 * 160
        min_face_size=20,
        thresholds=[0.6, 0.7, 0.7],
        factor=0.709,
        post_process=True,
        device=device,
        keep_all=True  # Return all faces, not just best one
    )
    
    # Load image
    img = Image.open(image_path).convert('RGB')
    print(f"   Image size: {img.size}")
    
    # Detect faces with bounding boxes
    boxes, probs = mtcnn.detect(img)
    
    if boxes is not None:
        print(f"   MTCNN detected: {len(boxes)} faces")
        for i, (box, prob) in enumerate(zip(boxes, probs)):
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            area_ratio = (w * h) / (img.size[0] * img.size[1])
            print(f"     Face {i+1}: ({x1:.0f},{y1:.0f},{w:.0f},{h:.0f}), prob={prob:.3f}, area_ratio={area_ratio:.6f}")
    else:
        print("   MTCNN: No faces detected")
    
    # Also test with keep_all=False (original preprocessing behavior)
    mtcnn_single = MTCNN(
        image_size=160,
        margin=32,
        min_face_size=20,
        thresholds=[0.6, 0.7, 0.7],
        factor=0.709,
        post_process=True,
        device=device,
        keep_all=False  # Only return best face
    )
    
    face_tensor = mtcnn_single(img)
    if face_tensor is not None:
        print(f"   MTCNN (single best): Successfully extracted 1 face")
    else:
        print(f"   MTCNN (single best): No face extracted")
    
    return boxes, probs

if __name__ == "__main__":
    image_path = "data/raw/personA/IMG_9569.png"
    
    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        exit(1)
    
    print(f"📷 Comparing detection methods on: {image_path}")
    print("=" * 60)
    
    # Test both methods
    opencv_faces = test_opencv_detection(image_path)
    mtcnn_boxes, mtcnn_probs = test_mtcnn_detection(image_path)
    
    print("\n📊 COMPARISON SUMMARY:")
    print(f"   OpenCV Haar Cascade: {len(opencv_faces)} faces")
    print(f"   MTCNN: {len(mtcnn_boxes) if mtcnn_boxes is not None else 0} faces")
    
    print("\n🔍 ANALYSIS:")
    print("   • MTCNN: Deep learning, fewer false positives, more accurate")
    print("   • OpenCV: Traditional CV, faster but more false positives")
    print("   • Server should use MTCNN for better accuracy!")