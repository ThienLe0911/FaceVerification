"""
Face Detection Debugger
Test và visualize face detection với bounding boxes
"""
import cv2
import face_recognition
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from pathlib import Path

def test_face_detection(image_path, output_dir="debug_output"):
    """
    Test face detection và tạo ảnh với bounding boxes
    """
    # Tạo thư mục output
    Path(output_dir).mkdir(exist_ok=True)
    
    print(f"🔍 Testing face detection cho: {image_path}")
    
    # Load ảnh với face_recognition (RGB format)
    image_rgb = face_recognition.load_image_file(image_path)
    
    # Load ảnh với OpenCV (BGR format) để visualize
    image_bgr = cv2.imread(image_path)
    image_display = image_bgr.copy()
    
    print(f"📐 Image size: {image_rgb.shape}")
    
    # Test với các model khác nhau
    models = ["hog", "cnn"]
    
    for model_name in models:
        print(f"\n🤖 Testing với model: {model_name}")
        
        try:
            # Face detection
            face_locations = face_recognition.face_locations(image_rgb, model=model_name)
            print(f"   Detected {len(face_locations)} khuôn mặt")
            
            # Create copy để vẽ
            annotated_image = image_display.copy()
            
            # Vẽ bounding boxes
            for i, (top, right, bottom, left) in enumerate(face_locations):
                print(f"   Face {i+1}: top={top}, right={right}, bottom={bottom}, left={left}")
                print(f"   Face {i+1}: width={right-left}, height={bottom-top}")
                
                # Vẽ rectangle
                cv2.rectangle(annotated_image, (left, top), (right, bottom), (0, 255, 0), 2)
                
                # Vẽ label
                cv2.putText(annotated_image, f"Face {i+1}", 
                           (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.6, (0, 255, 0), 2)
                
                # Vẽ confidence area (để debug)
                center_x = (left + right) // 2
                center_y = (top + bottom) // 2
                cv2.circle(annotated_image, (center_x, center_y), 3, (255, 0, 0), -1)
            
            # Save annotated image
            filename = Path(image_path).stem
            output_path = Path(output_dir) / f"{filename}_{model_name}_detection.jpg"
            cv2.imwrite(str(output_path), annotated_image)
            print(f"   💾 Saved: {output_path}")
            
            # Generate embeddings cho detected faces
            if face_locations:
                print(f"   🧠 Generating embeddings...")
                face_encodings = face_recognition.face_encodings(image_rgb, face_locations)
                print(f"   Generated {len(face_encodings)} embeddings")
                
                for i, encoding in enumerate(face_encodings):
                    print(f"   Embedding {i+1}: shape={encoding.shape}, norm={np.linalg.norm(encoding):.4f}")
            
        except Exception as e:
            print(f"   ❌ Error với model {model_name}: {e}")
    
    return face_locations

def compare_detection_methods(image_path):
    """
    So sánh các phương pháp detection khác nhau
    """
    print(f"\n🆚 So sánh detection methods cho: {image_path}")
    
    image = face_recognition.load_image_file(image_path)
    
    # Test các số lần upsampling khác nhau
    upsample_values = [0, 1, 2]
    
    for upsample in upsample_values:
        print(f"\n📈 Upsampling = {upsample}")
        
        for model in ["hog", "cnn"]:
            try:
                face_locations = face_recognition.face_locations(
                    image, 
                    number_of_times_to_upsample=upsample,
                    model=model
                )
                print(f"   {model.upper()}: {len(face_locations)} faces")
            except Exception as e:
                print(f"   {model.upper()}: Error - {e}")

def test_with_opencv_cascade(image_path, output_dir="debug_output"):
    """
    Test với OpenCV Haar Cascade để so sánh
    """
    print(f"\n👁️ Testing với OpenCV Haar Cascade")
    
    # Load ảnh
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Load Haar cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    print(f"   OpenCV detected {len(faces)} faces")
    
    # Draw rectangles
    annotated_image = image.copy()
    for i, (x, y, w, h) in enumerate(faces):
        cv2.rectangle(annotated_image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(annotated_image, f"OpenCV Face {i+1}", 
                   (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, (255, 0, 0), 2)
        print(f"   Face {i+1}: x={x}, y={y}, w={w}, h={h}")
    
    # Save result
    filename = Path(image_path).stem
    output_path = Path(output_dir) / f"{filename}_opencv_detection.jpg"
    cv2.imwrite(str(output_path), annotated_image)
    print(f"   💾 Saved: {output_path}")

if __name__ == "__main__":
    # Test với ảnh cụ thể
    image_path = "/Users/thienlehoang/studyAnything/projectThiGiacMayTinh/face_verification_project/data/raw/query_images/single_face/IMG_9703.jpg"
    
    if os.path.exists(image_path):
        print("🚀 Starting Face Detection Debug...")
        
        # Test face_recognition
        face_locations = test_face_detection(image_path)
        
        # So sánh methods
        compare_detection_methods(image_path)
        
        # Test OpenCV
        test_with_opencv_cascade(image_path)
        
        print("\n✅ Debug hoàn thành! Kiểm tra thư mục debug_output/")
    else:
        print(f"❌ File không tồn tại: {image_path}")