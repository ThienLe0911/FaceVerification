# Hướng Dẫn Sử Dụng Face Verification System

## 🎯 Tổng Quan

Hệ thống Face Verification hỗ trợ **2 luồng chính**:

### A. **Enroll Flow** (Tạo Gallery PersonA)
- Upload 20-40 ảnh PersonA để tạo gallery
- Hệ thống phân tích chất lượng và đa dạng
- Tạo gallery khi đạt điều kiện

### B. **Verify Flow** (Kiểm Tra PersonA)
- Upload ảnh cần kiểm tra
- Hệ thống phát hiện và so sánh với gallery
- Trả về verdict có PersonA hay không

---

## 🚀 Cách Chạy Hệ Thống

### 1. Khởi Động Backend
```bash
cd server
python3 simple_server.py
```
**Backend chạy tại:** http://localhost:8000

### 2. Khởi Động Frontend
```bash
cd web
npm run dev
```
**Frontend chạy tại:** http://localhost:3001

### 3. Truy Cập Ứng Dụng
- **Web Interface:** http://localhost:3001
- **API Documentation:** http://localhost:8000/docs

---

## 📖 UX Flow Chi Tiết

### 🔧 Enroll Flow (Khuyến nghị thực hiện trước)

#### Bước 1: Vào Enroll Page
- Click "Enroll PersonA" trên header
- Đọc hướng dẫn: "Upload 20-40 ảnh PersonA"

#### Bước 2: Upload Ảnh
- **Cách 1:** Click "Click để chọn ảnh" → chọn nhiều ảnh
- **Cách 2:** Drag & drop ảnh vào vùng upload
- **Lưu ý:** Tối đa 40 ảnh, mỗi ảnh < 10MB

#### Bước 3: Xem Kết Quả Xử Lý
- Backend detect & crop faces
- Hiển thị list ảnh OK/lỗi:
  - ✅ **Success:** 1 face detected, quality score
  - ❌ **No face:** Không phát hiện face
  - ❌ **Multiple faces:** Nhiều faces trong ảnh

#### Bước 4: Kiểm Tra Gallery Stats
**Panel bên phải hiển thị:**
- **Số ảnh:** x/40 (progress bar)
- **Diversity Score:** y/100 (đa dạng góc độ)
- **Quality Score:** z/100 (chất lượng trung bình)
- **Status:** Empty/Insufficient/Ready/Excellent

#### Bước 5: Đọc Recommendations
- Nếu **count < 15** → "Cần thêm X ảnh nữa"
- Nếu **diversity < 65** → "Thêm ảnh góc độ khác"
- Nếu **quality < 50** → "Thêm ảnh chất lượng cao"

#### Bước 6: Tạo Gallery (khi sẵn sàng)
- Điều kiện: count ≥ 15 và quality ≥ 50
- Click "Tạo Gallery PersonA"
- Backend chạy enroll.py → lưu personA_normalized.npz
- Hiển thị ✅ "Gallery PersonA đã được tạo thành công"

---

### 🔍 Verify Flow

#### Bước 1: Vào Verify Page
- Click "Verify PersonA" trên header
- Đọc hướng dẫn về verify flow

#### Bước 2: Upload Ảnh Test
- **Single image hoặc batch**
- Drag & drop hoặc click upload
- Preview ảnh gốc

#### Bước 3: Chạy Verification
- Click "Verify PersonA"
- Backend detect faces → embed → compare with gallery mean

#### Bước 4: Xem Kết Quả Chi Tiết

**Verdict Panel:**
- 🟢 **"Có PersonA trong ảnh"** + confidence score
- 🔴 **"Không tìm thấy PersonA"** + confidence score
- Thống kê: Tổng faces / PersonA detected

**Face Details Table:**
```
| Face ID | Predicted | Similarity | Confidence | Status |
|---------|-----------|------------|------------|---------|
| #1      | PersonA   | 87.3%      | 91.5%      | ✅      |
| #2      | Unknown   | 34.2%      | 38.1%      | ❌      |
```

**Annotated Image:**
- Ảnh có bounding boxes và labels
- Button "Download" để tải ảnh kết quả

#### Bước 5: Đọc Suggestions
- **Nếu borderline:** "Vui lòng thử với ảnh khác"
- **Nếu chưa có gallery:** "Chưa có gallery PersonA. Vui lòng enroll trước"

---

## ⚙️ Settings Page

### Điều Chỉnh Threshold
- **Slider 0.1 - 0.9**
- **Mô tả real-time:**
  - 0.1-0.3: Very permissive (có thể accept sai)
  - 0.4-0.6: Balanced (khuyến nghị)
  - 0.7-0.9: Very strict (có thể reject đúng)

### Lưu Cài Đặt
- Click "Save Changes"
- Áp dụng ngay cho verify flow

---

## 🎨 UI Features

### Header Navigation
- **Logo:** Face Verification + PersonA Recognition System
- **Tabs:** Enroll PersonA | Verify PersonA | Cài đặt
- **Sub-nav:** Enroll: Tạo gallery → Verify: Kiểm tra ảnh

### Responsive Design
- **Desktop:** Full layout với panels
- **Mobile:** Stack layout, touch-friendly
- **Tablet:** Optimized columns

### Visual Feedback
- **Progress bars** cho tất cả metrics
- **Color coding:**
  - 🟢 Green: Success/Ready
  - 🟡 Yellow: Warning/Insufficient  
  - 🔴 Red: Error/Failed
  - 🔵 Blue: Info/Processing
- **Icons** rõ ràng cho mọi action
- **Toast notifications** cho user feedback

---

## 🔧 Technical Details

### API Endpoints
- `POST /api/enroll/batch` - Upload multiple images
- `POST /api/enroll/create-gallery` - Create PersonA gallery
- `GET /api/enroll/stats` - Get gallery statistics
- `POST /api/verify` - Verify faces in image
- `GET/POST /api/threshold` - Manage threshold

### File Handling
- **Formats:** JPG, PNG, GIF, WebP
- **Max size:** 10MB per file
- **Max batch:** 40 files for enroll

### Quality Scoring
- **Face detection:** Single face preferred
- **Image quality:** Resolution, lighting, blur
- **Diversity:** Different angles, expressions
- **Gallery readiness:** Combination of above

---

## 🏃‍♂️ Quick Start Checklist

1. ✅ **Chạy backend:** `cd server && python3 simple_server.py`
2. ✅ **Chạy frontend:** `cd web && npm run dev`
3. ✅ **Truy cập:** http://localhost:3001
4. ✅ **Enroll flow:** Upload 20-40 ảnh PersonA
5. ✅ **Tạo gallery:** Khi stats đủ điều kiện
6. ✅ **Verify flow:** Upload ảnh test và xem kết quả

**🎉 Bây giờ bạn có thể sử dụng hệ thống Face Verification hoàn chỉnh!**