#!/usr/bin/env python3
"""
Test Bounding Box Visualization
Kiểm tra visualization với ảnh thật
"""

import cv2
import numpy as np
from pathlib import Path

def test_opencv_annotation():
    """Test OpenCV annotation với demo data"""
    print("🎨 Testing OpenCV Annotation...")
    
    # Tạo ảnh demo
    width, height = 640, 480
    image = np.ones((height, width, 3), dtype=np.uint8) * 240  # Light gray
    
    # Demo faces data
    faces_data = [
        {
            "face_id": 1,
            "bbox": [100, 100, 200, 200],
            "similarity": 0.85,
            "is_personA": True
        },
        {
            "face_id": 2, 
            "bbox": [300, 150, 400, 250],
            "similarity": 0.35,
            "is_personA": False
        },
        {
            "face_id": 3,
            "bbox": [450, 80, 550, 180],
            "similarity": 0.72,
            "is_personA": True
        }
    ]
    
    verification_result = {
        "verdict": "Có PersonA trong ảnh",
        "confidence": 0.85
    }
    
    # Vẽ background pattern
    for i in range(0, width, 50):
        cv2.line(image, (i, 0), (i, height), (220, 220, 220), 1)
    for i in range(0, height, 50):
        cv2.line(image, (0, i), (width, i), (220, 220, 220), 1)
    
    # Vẽ bounding boxes
    for i, face_data in enumerate(faces_data):
        bbox = face_data.get('bbox', [])
        similarity = face_data.get('similarity', 0)
        is_personA = face_data.get('is_personA', False)
        
        if len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            print(f"   Face {i+1}: bbox=({x1},{y1},{x2},{y2}), similarity={similarity:.3f}")
            
            # Màu sắc: xanh = PersonA, đỏ = Unknown
            color = (0, 255, 0) if is_personA else (0, 0, 255)  # BGR format
            thickness = 3
            
            # Vẽ rectangle
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
            
            # Tạo label
            if is_personA:
                label = f"PersonA {similarity:.2f}"
            else:
                label = f"Unknown {similarity:.2f}"
            
            # Vẽ background cho text
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
            )
            cv2.rectangle(
                image, 
                (int(x1), int(y1) - text_height - 10),
                (int(x1) + text_width, int(y1)), 
                color, -1
            )
            
            # Vẽ text
            cv2.putText(
                image, label,
                (int(x1), int(y1) - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
            )
            
            # Vẽ confidence bar
            bar_width = int((x2 - x1) * similarity)
            cv2.rectangle(
                image,
                (int(x1), int(y2) + 5),
                (int(x1) + bar_width, int(y2) + 15),
                color, -1
            )
    
    # Thêm thông tin tổng quan
    if verification_result:
        verdict = verification_result.get('verdict', '')
        confidence = verification_result.get('confidence', 0)
        
        # Vẽ verdict ở góc trên
        verdict_color = (0, 255, 0) if 'Có PersonA' in verdict else (0, 0, 255)
        cv2.rectangle(image, (10, 10), (400, 60), verdict_color, -1)
        cv2.putText(image, verdict, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(image, f"Confidence: {confidence:.1%}", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Save test image
    output_path = Path("test_bbox_demo.jpg")
    success = cv2.imwrite(str(output_path), image)
    
    if success:
        print(f"✅ Demo annotation saved: {output_path}")
        print(f"   Image size: {width}x{height}")
        print(f"   Faces: {len(faces_data)}")
        return True
    else:
        print("❌ Failed to save demo")
        return False

def test_real_image_annotation(image_path: str):
    """Test annotation với ảnh thật"""
    print(f"\n📷 Testing với ảnh thật: {image_path}")
    
    image_file = Path(image_path)
    if not image_file.exists():
        print(f"❌ File không tồn tại: {image_path}")
        return False
    
    try:
        # Load ảnh
        image = cv2.imread(str(image_file))
        if image is None:
            print("❌ Không thể load ảnh với OpenCV")
            return False
        
        height, width = image.shape[:2]
        print(f"   Original size: {width}x{height}")
        
        # Demo faces data (tương tự như server sẽ detect)
        faces_data = [
            {
                "face_id": 1,
                "bbox": [width//4, height//4, width//2, height//2],
                "similarity": 0.75,
                "is_personA": True
            }
        ]
        
        # Vẽ bounding boxes
        for i, face_data in enumerate(faces_data):
            bbox = face_data.get('bbox', [])
            similarity = face_data.get('similarity', 0)
            is_personA = face_data.get('is_personA', False)
            
            if len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                
                color = (0, 255, 0) if is_personA else (0, 0, 255)
                thickness = max(3, min(width, height) // 200)  # Responsive thickness
                
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
                
                label = f"PersonA {similarity:.2f}" if is_personA else f"Unknown {similarity:.2f}"
                font_scale = max(0.5, min(width, height) / 1000)  # Responsive font
                
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2
                )
                cv2.rectangle(
                    image, 
                    (int(x1), int(y1) - text_height - 10),
                    (int(x1) + text_width, int(y1)), 
                    color, -1
                )
                
                cv2.putText(
                    image, label,
                    (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2
                )
        
        # Save annotated image
        output_path = Path(f"annotated_test_{image_file.stem}.jpg")
        success = cv2.imwrite(str(output_path), image)
        
        if success:
            print(f"✅ Annotated image saved: {output_path}")
            return True
        else:
            print("❌ Failed to save annotated image")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    import sys
    print("🔍 Testing Bounding Box Visualization")
    print("=" * 50)
    
    # Test 1: Demo annotation
    success1 = test_opencv_annotation()
    
    # Test 2: Real image (check command line arguments first)
    success2 = False
    
    if len(sys.argv) > 1:
        # Command line argument provided
        img_path = sys.argv[1]
        if Path(img_path).exists():
            success2 = test_real_image_annotation(img_path)
        else:
            print(f"❌ File không tồn tại: {img_path}")
    else:
        # Auto-detect test images
        test_images = [
            "../github/IMG_9703.jpg",
            "IMG_9703.jpg", 
            "test_image.jpg"
        ]
        
        for img_path in test_images:
            if Path(img_path).exists():
                print(f"📷 Found test image: {img_path}")
                success2 = test_real_image_annotation(img_path)
                break
    
    if not success2 and len(sys.argv) <= 1:
        print("\n📝 Không tìm thấy ảnh test. Tạo ảnh test...")
        print("   Sử dụng: python test_bbox_visualization.py <path_to_image>")
    
    print(f"\n📊 Results:")
    print(f"   Demo annotation: {'✅' if success1 else '❌'}")
    print(f"   Real image test: {'✅' if success2 else '📝 Skipped'}")
    
    if success1:
        print(f"\n🎉 OpenCV annotation working! Ready for server integration.")