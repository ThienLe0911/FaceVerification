"""
Simple FastAPI Server cho Face Verification System
Hỗ trợ 2 luồng chính với demo functionality
"""

import os
import json
import time
import shutil
import random
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    from fastapi import FastAPI, File, UploadFile, HTTPException, Form
    from fastapi.staticfiles import StaticFiles
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
    import uvicorn
    from PIL import Image
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    print("⚠️ FastAPI dependencies không có sẵn. Chạy: pip install fastapi uvicorn python-multipart pillow")
    DEPENDENCIES_AVAILABLE = False

if DEPENDENCIES_AVAILABLE:
    # Initialize FastAPI app
    app = FastAPI(
        title="Face Verification API",
        description="API cho Enroll và Verify flow",
        version="2.0.0"
    )

    # Enable CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://localhost:3001", "http://localhost:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Tạo thư mục cần thiết
    BASE_DIR = Path(__file__).parent
    UPLOAD_DIR = BASE_DIR / "uploads"
    STATIC_DIR = BASE_DIR / "static"
    ANNOTATIONS_DIR = STATIC_DIR / "annotations"
    GALLERY_DIR = STATIC_DIR / "gallery"

    for directory in [UPLOAD_DIR, STATIC_DIR, ANNOTATIONS_DIR, GALLERY_DIR]:
        directory.mkdir(exist_ok=True)

    # Mount static files
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # Global state
    gallery_data = {
        "images": [],  # List of gallery images
        "count": 0,
        "diversity_score": 0,
        "quality_score": 0,
        "status": "empty"  # empty, insufficient, ready
    }
    verification_threshold = 0.5

    # Pydantic models
    class EnrollBatchResponse(BaseModel):
        status: str
        message: str
        processed_images: List[Dict[str, Any]]
        gallery_stats: Dict[str, Any]
        suggestions: List[str]

    class VerifyResponse(BaseModel):
        status: str
        total_faces: int
        faces: List[Dict[str, Any]]
        annotated_image_path: Optional[str]
        verdict: str
        confidence: float
        suggestions: List[str]

    class ThresholdResponse(BaseModel):
        threshold: float
        message: str

    class GalleryStatsResponse(BaseModel):
        count: int
        diversity_score: int
        quality_score: int
        status: str
        recommendations: List[str]

    # Utility functions
    def is_valid_image(file: UploadFile) -> bool:
        """Kiểm tra file có phải ảnh hợp lệ không"""
        valid_types = ["image/jpeg", "image/png", "image/gif", "image/webp"]
        return file.content_type in valid_types

    def save_uploaded_file(file: UploadFile, filename: str, directory: Path) -> Path:
        """Lưu file được upload"""
        file_path = directory / filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return file_path

    def simulate_face_detection(image_path: Path) -> Dict[str, Any]:
        """Detect faces using OpenCV Haar Cascade (more reliable than face_recognition)"""
        try:
            print(f"🔍 Face detection for {image_path.name}")
            
            # Try OpenCV face detection first
            try:
                import cv2
                
                # Initialize cascade if not already done
                if not hasattr(simulate_face_detection, 'cascade'):
                    simulate_face_detection.cascade = cv2.CascadeClassifier(
                        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                    )
                
                cascade = simulate_face_detection.cascade
                if cascade.empty():
                    raise Exception("Could not load face cascade")
                
                # Load and process image
                image = cv2.imread(str(image_path))
                if image is None:
                    raise Exception("Could not load image")
                
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
                
                print(f"   OpenCV detected {len(faces)} raw faces")
                
                # Filter and validate faces
                valid_faces = []
                for i, (x, y, w, h) in enumerate(faces):
                    face_area = w * h
                    image_area = width * height
                    area_ratio = face_area / image_area
                    
                    print(f"   Face {i+1}: size={w}x{h}, area_ratio={area_ratio:.4f}")
                    
                    # Quality scoring based on size - Made less strict for smaller faces
                    if area_ratio < 0.003:  # Changed from 0.01 to 0.003 (0.3% instead of 1%)
                        print(f"     -> Too small (area_ratio={area_ratio:.4f}), filtered out")
                        continue
                    elif area_ratio > 0.5:
                        print(f"     -> Too large, filtered out")
                        continue
                    
                    # Good face - Updated scoring ranges
                    if area_ratio > 0.15:
                        quality = random.randint(80, 95)
                    elif area_ratio > 0.02:  # Changed from 0.05 to 0.02
                        quality = random.randint(70, 85)
                    elif area_ratio > 0.005:  # New range for small but valid faces
                        quality = random.randint(55, 70)
                    else:
                        quality = random.randint(45, 60)  # Small faces still get some chance
                    
                    face_data = {
                        "x": int(x), "y": int(y), "w": int(w), "h": int(h),
                        "bbox": [int(x), int(y), int(x + w), int(y + h)],
                        "area_ratio": area_ratio,
                        "quality": quality
                    }
                    
                    valid_faces.append(face_data)
                    print(f"     -> Valid face, quality={quality}")
                
                print(f"   ✅ Final: {len(valid_faces)} valid faces")
                
                # Return results based on face count
                if len(valid_faces) == 0:
                    return {
                        "status": "no_face",
                        "message": "Không phát hiện khuôn mặt",
                        "faces_count": 0,
                        "quality": 0,
                        "face_locations": []
                    }
                elif len(valid_faces) == 1:
                    face = valid_faces[0]
                    return {
                        "status": "success", 
                        "message": "Phát hiện 1 khuôn mặt",
                        "faces_count": 1,
                        "quality": face["quality"],
                        "face_locations": [(face["y"], face["x"] + face["w"], face["y"] + face["h"], face["x"])],  # (top, right, bottom, left)
                        "bbox": face["bbox"]
                    }
                else:
                    # Multiple faces detected
                    avg_quality = sum(f["quality"] for f in valid_faces) // len(valid_faces)
                    face_locations = []
                    for face in valid_faces:
                        face_locations.append((face["y"], face["x"] + face["w"], face["y"] + face["h"], face["x"]))
                    
                    return {
                        "status": "multiple_faces",
                        "message": f"Phát hiện {len(valid_faces)} khuôn mặt", 
                        "faces_count": len(valid_faces),
                        "quality": avg_quality,
                        "face_locations": face_locations
                    }
                
            except ImportError:
                print("⚠️ OpenCV not available, using simulation")
            except Exception as e:
                print(f"⚠️ OpenCV error: {e}, fallback to simulation")
            
            # Fallback: Original simulation code
            img = Image.open(image_path)
            width, height = img.size
            
            face_detected = random.choice([True, True, True, False])  # 75% có face
            multiple_faces = random.choice([False, False, False, True])  # 25% có nhiều face
            
            if not face_detected:
                return {
                    "status": "no_face",
                    "message": "Không phát hiện khuôn mặt",
                    "faces_count": 0,
                    "quality": 0,
                    "face_locations": []
                }
            elif multiple_faces:
                # Simulate multiple faces
                num_faces = random.randint(2, 4)
                fake_locations = []
                for i in range(num_faces):
                    top = random.randint(10, height//2)
                    left = random.randint(10, width//2)
                    bottom = top + random.randint(50, 150)
                    right = left + random.randint(50, 150)
                    fake_locations.append((top, right, bottom, left))
                
                return {
                    "status": "multiple_faces", 
                    "message": f"Phát hiện {num_faces} khuôn mặt",
                    "faces_count": num_faces,
                    "quality": 30,
                    "face_locations": fake_locations
                }
            else:
                # Good single face
                quality = random.randint(70, 95)
                top = random.randint(50, height//3)
                left = random.randint(50, width//3)
                bottom = top + random.randint(100, 200)
                right = left + random.randint(100, 200)
                
                return {
                    "status": "success",
                    "message": "Phát hiện 1 khuôn mặt",
                    "faces_count": 1,
                    "quality": quality,
                    "face_locations": [(top, right, bottom, left)],
                    "bbox": [left, top, right, bottom]
                }
                
        except Exception as e:
            return {
                "status": "error",
                "message": f"Lỗi xử lý ảnh: {str(e)}",
                "faces_count": 0,
                "quality": 0,
                "face_locations": []
            }

    def calculate_gallery_stats() -> Dict[str, Any]:
        """Tính toán statistics cho gallery"""
        count = len(gallery_data["images"])
        
        if count == 0:
            diversity = quality = 0
            status = "empty"
            recommendations = ["Thêm ít nhất 15-20 ảnh PersonA để bắt đầu"]
        elif count < 15:
            diversity = min(count * 4, 60)  # Max 60 cho < 15 ảnh
            quality = sum(img.get("quality", 0) for img in gallery_data["images"]) / count
            status = "insufficient"
            recommendations = [
                f"Cần thêm {15 - count} ảnh nữa (tối thiểu 15)",
                "Thêm ảnh với góc độ khác nhau",
                "Thêm ảnh với điều kiện ánh sáng khác nhau"
            ]
        elif count < 30:
            diversity = min(60 + (count - 15) * 2, 85)
            quality = sum(img.get("quality", 0) for img in gallery_data["images"]) / count
            if diversity >= 65 and quality >= 70:
                status = "ready"
                recommendations = ["Gallery đã sẵn sàng! Có thể tạo gallery hoặc thêm ảnh để cải thiện độ chính xác"]
            else:
                status = "insufficient"
                recommendations = [
                    "Thêm ảnh với các góc độ khác nhau để tăng diversity",
                    "Thêm ảnh chất lượng cao hơn"
                ]
        else:
            diversity = min(85 + (count - 30), 95)
            quality = sum(img.get("quality", 0) for img in gallery_data["images"]) / count
            status = "excellent"
            recommendations = ["Gallery rất tốt! Sẵn sàng để sử dụng"]

        return {
            "count": count,
            "diversity_score": int(diversity),
            "quality_score": int(quality),
            "status": status,
            "recommendations": recommendations
        }

    def create_annotated_image(image_path: Path, faces_data: List[Dict], verification_result=None) -> Path:
        """Tạo ảnh có annotation với bounding boxes thật sự"""
        try:
            # Try to use OpenCV for real annotation
            try:
                import cv2
                print(f"🎨 Tạo annotated image cho {image_path.name}")
                
                # Load image
                image = cv2.imread(str(image_path))
                if image is None:
                    raise Exception("Không thể load ảnh với OpenCV")
                
                height, width = image.shape[:2]
                print(f"   Image size: {width}x{height}")
                
                # Vẽ bounding boxes cho từng face
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
                
                # Save annotated image
                timestamp = int(time.time())
                annotation_filename = f"annotated_{timestamp}_{image_path.stem}.jpg"
                annotation_path = ANNOTATIONS_DIR / annotation_filename
                
                success = cv2.imwrite(str(annotation_path), image)
                if success:
                    print(f"   💾 Saved annotated image: {annotation_path}")
                    return annotation_path
                else:
                    raise Exception("Lỗi save ảnh")
                    
            except ImportError:
                print("⚠️ OpenCV không có sẵn, tạo annotation đơn giản")
                # Fallback: chỉ copy file gốc
                pass
            
            # Fallback: copy original image
            timestamp = int(time.time())
            annotation_filename = f"demo_annotation_{timestamp}.jpg"
            annotation_path = ANNOTATIONS_DIR / annotation_filename
            
            # Copy original image
            img = Image.open(image_path)
            img.save(annotation_path, "JPEG", quality=85)
            print(f"   📋 Copied original as annotation: {annotation_path}")
            
            return annotation_path
            
        except Exception as e:
            print(f"❌ Error creating annotation: {e}")
            return None

    # API Endpoints
    @app.get("/")
    async def root():
        return {
            "message": "Face Verification API v2.0",
            "flows": ["enroll", "verify"],
            "docs": "/docs"
        }

    @app.post("/api/enroll/batch", response_model=EnrollBatchResponse)
    async def enroll_batch(files: List[UploadFile] = File(...)):
        """
        Enroll Flow: Upload batch ảnh PersonA
        """
        if len(files) > 50:
            raise HTTPException(status_code=400, detail="Tối đa 50 ảnh một lần")
        
        processed_images = []
        suggestions = []
        
        for file in files:
            if not is_valid_image(file):
                processed_images.append({
                    "filename": file.filename,
                    "status": "invalid_format",
                    "message": "Định dạng file không hỗ trợ"
                })
                continue
            
            if file.size > 10 * 1024 * 1024:
                processed_images.append({
                    "filename": file.filename,
                    "status": "too_large", 
                    "message": "File quá lớn (> 10MB)"
                })
                continue
            
            try:
                # Save file
                timestamp = int(time.time())
                filename = f"gallery_{timestamp}_{file.filename}"
                image_path = save_uploaded_file(file, filename, GALLERY_DIR)
                
                # Detect face
                detection_result = simulate_face_detection(image_path)
                
                if detection_result["status"] == "success":
                    # Add to gallery
                    image_data = {
                        "filename": filename,
                        "original_name": file.filename,
                        "path": str(image_path),
                        "quality": detection_result["quality"],
                        "bbox": detection_result.get("bbox"),
                        "added_at": timestamp
                    }
                    gallery_data["images"].append(image_data)
                    
                    processed_images.append({
                        "filename": file.filename,
                        "status": "success",
                        "message": f"Đã thêm vào gallery (Quality: {detection_result['quality']})",
                        "quality": detection_result["quality"]
                    })
                else:
                    processed_images.append({
                        "filename": file.filename,
                        "status": detection_result["status"],
                        "message": detection_result["message"]
                    })
                    # Xóa file lỗi
                    if image_path.exists():
                        image_path.unlink()
                
            except Exception as e:
                processed_images.append({
                    "filename": file.filename,
                    "status": "error",
                    "message": f"Lỗi xử lý: {str(e)}"
                })
        
        # Calculate gallery stats
        stats = calculate_gallery_stats()
        gallery_data.update(stats)
        
        # Generate suggestions based on results
        success_count = len([img for img in processed_images if img["status"] == "success"])
        
        if success_count == 0:
            suggestions.append("Không có ảnh nào được xử lý thành công. Kiểm tra chất lượng ảnh")
        elif stats["status"] == "insufficient":
            suggestions.extend(stats["recommendations"])
        
        return EnrollBatchResponse(
            status="completed",
            message=f"Đã xử lý {len(files)} ảnh, thành công {success_count}",
            processed_images=processed_images,
            gallery_stats=stats,
            suggestions=suggestions
        )

    @app.post("/api/enroll/create-gallery")
    async def create_gallery():
        """Tạo gallery từ các ảnh đã upload"""
        stats = calculate_gallery_stats()
        
        if stats["count"] < 15:
            raise HTTPException(
                status_code=400,
                detail=f"Cần ít nhất 15 ảnh để tạo gallery (hiện có {stats['count']})"
            )
        
        if stats["quality_score"] < 50:
            raise HTTPException(
                status_code=400,
                detail=f"Chất lượng gallery quá thấp ({stats['quality_score']}/100). Thêm ảnh chất lượng cao hơn"
            )
        
        # Simulate gallery creation
        time.sleep(2)  # Simulate processing
        
        return {
            "status": "success",
            "message": "Gallery PersonA đã được tạo thành công",
            "stats": stats,
            "gallery_path": "/static/gallery/personA_normalized.npz"
        }

    @app.get("/api/enroll/stats", response_model=GalleryStatsResponse)
    async def get_gallery_stats():
        """Lấy thống kê gallery hiện tại"""
        stats = calculate_gallery_stats()
        return GalleryStatsResponse(**stats)

    @app.delete("/api/enroll/reset")
    async def reset_gallery():
        """Xóa toàn bộ gallery"""
        gallery_data["images"].clear()
        # Xóa các file
        for file_path in GALLERY_DIR.glob("gallery_*"):
            file_path.unlink()
        
        return {"status": "success", "message": "Đã xóa toàn bộ gallery"}

    @app.post("/api/verify", response_model=VerifyResponse)
    async def verify_image(file: UploadFile = File(...)):
        """
        Verify Flow: Kiểm tra ảnh có PersonA không
        """
        if not is_valid_image(file):
            raise HTTPException(status_code=400, detail="Định dạng ảnh không hỗ trợ")
        
        if file.size > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File quá lớn")
        
        try:
            # Save file
            timestamp = int(time.time())
            filename = f"verify_{timestamp}_{file.filename}"
            image_path = save_uploaded_file(file, filename, UPLOAD_DIR)
            
            # Real face detection
            detection_result = simulate_face_detection(image_path)
            print(f"🔍 Detection result: {detection_result}")
            
            if detection_result["status"] == "error":
                raise HTTPException(status_code=500, detail=f"Lỗi face detection: {detection_result['message']}")
            elif detection_result["status"] == "no_face":
                return VerifyResponse(
                    status="no_face", 
                    total_faces=0,
                    faces=[],
                    annotated_image_path="/static/no-face.jpg",
                    verdict="Không phát hiện khuôn mặt",
                    confidence=0.0,
                    suggestions=["Thử với ảnh có khuôn mặt rõ ràng hơn"]
                )
            
            # Convert detection result to verification format
            faces = []
            for i in range(detection_result["faces_count"]):
                # Simulate face comparison with gallery
                has_personA = random.choice([True, False]) if gallery_data["count"] > 0 else False
                similarity = random.uniform(0.3, 0.95) if has_personA else random.uniform(0.1, 0.4)
                
                # Use real bbox from detection or fallback to fake
                if "bbox" in detection_result:
                    bbox = detection_result["bbox"]
                elif detection_result.get("face_locations"):
                    # Convert (top, right, bottom, left) to [x1, y1, x2, y2]
                    top, right, bottom, left = detection_result["face_locations"][i] if i < len(detection_result["face_locations"]) else detection_result["face_locations"][0]
                    bbox = [left, top, right, bottom]
                else:
                    # Fallback to fake bbox
                    bbox = [50 + i * 100, 50 + i * 50, 150 + i * 100, 150 + i * 50]
                
                face_data = {
                    "face_id": i + 1,
                    "bbox": bbox,
                    "similarity": round(similarity, 3),
                    "predicted": "PersonA" if similarity >= verification_threshold else "Unknown",
                    "is_personA": similarity >= verification_threshold,
                    "confidence": min(similarity * 1.2, 0.99) if similarity >= verification_threshold else similarity
                }
                faces.append(face_data)
            
            # Total faces detected
            total_faces = len(faces)
            
            # Overall verdict (calculate first)
            personA_faces = [f for f in faces if f["is_personA"]]
            if personA_faces:
                verdict = "Có PersonA trong ảnh"
                confidence = max(f["confidence"] for f in personA_faces)
                suggestions = []
            else:
                verdict = "Không tìm thấy PersonA"
                confidence = max((f["similarity"] for f in faces), default=0)
                if confidence > 0.3:
                    suggestions = ["Kết quả không chắc chắn. Vui lòng thử với ảnh khác"]
                else:
                    suggestions = []
            
            if gallery_data["count"] == 0:
                suggestions.append("Chưa có gallery PersonA. Vui lòng enroll trước")
            
            # Create annotated image with real bounding boxes
            verification_result = {
                "verdict": verdict,
                "confidence": confidence
            }
            
            annotation_path_obj = create_annotated_image(image_path, faces, verification_result)
            if annotation_path_obj:
                annotation_path = f"/static/annotations/{annotation_path_obj.name}"
            else:
                # Fallback to simple copy
                annotation_filename = f"annotated_{timestamp}.jpg"
                annotation_path = f"/static/annotations/{annotation_filename}"
                shutil.copy2(image_path, ANNOTATIONS_DIR / annotation_filename)
            
            return VerifyResponse(
                status="success",
                total_faces=total_faces,
                faces=faces,
                annotated_image_path=annotation_path,
                verdict=verdict,
                confidence=confidence,
                suggestions=suggestions
            )
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Lỗi verify: {str(e)}")

    @app.get("/api/threshold", response_model=ThresholdResponse)
    async def get_threshold():
        return ThresholdResponse(
            threshold=verification_threshold,
            message=f"Ngưỡng verification hiện tại: {verification_threshold}"
        )

    @app.post("/api/threshold", response_model=ThresholdResponse)
    async def set_threshold(threshold: float = Form(...)):
        global verification_threshold
        
        if not 0.1 <= threshold <= 0.9:
            raise HTTPException(status_code=400, detail="Ngưỡng phải từ 0.1 đến 0.9")
        
        verification_threshold = threshold
        return ThresholdResponse(
            threshold=verification_threshold,
            message=f"Đã cập nhật ngưỡng: {verification_threshold}"
        )

    @app.get("/api/status")
    async def get_status():
        """Lấy trạng thái hệ thống"""
        return {
            "gallery": calculate_gallery_stats(),
            "verification_threshold": verification_threshold,
            "system": "demo_mode"
        }

if __name__ == "__main__":
    if not DEPENDENCIES_AVAILABLE:
        print("Không thể khởi động server. Cài đặt dependencies:")
        print("pip install fastapi uvicorn python-multipart pillow")
    else:
        print("🚀 Khởi động Face Verification Server v2.0...")
        print("📱 API Docs: http://localhost:8000/docs")
        print("🔧 Enroll Flow: Upload nhiều ảnh PersonA")
        print("🔍 Verify Flow: Kiểm tra ảnh có PersonA không")
        
        uvicorn.run(
            "simple_server:app",
            host="0.0.0.0",
            port=8000,
            reload=True
        )