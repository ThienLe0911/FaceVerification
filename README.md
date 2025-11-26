# Face Verification System 🔐

A complete face verification system with a modern web interface, featuring face enrollment, multi-face detection, and verification with configurable thresholds. **Achieving 100% accuracy** on enrolled faces with an optimized production-ready pipeline.

## 🌟 Features

### Core Functionality
- **Face Enrollment**: Register new people with automatic face detection
- **Multi-Face Verification**: Detect and verify multiple faces in a single image  
- **Configurable Threshold**: Adjust verification sensitivity in real-time
- **High Accuracy**: Optimized for 100% accuracy on enrolled faces
- **Production Ready**: Robust pipeline with comprehensive error handling

### Web Interface 🎨
- **Modern React UI**: Responsive, intuitive interface
- **Drag & Drop Upload**: Easy file handling with preview
- **Real-time Results**: Instant feedback with visual annotations
- **Settings Management**: Interactive threshold configuration
- **Mobile Friendly**: Works on all devices

### API Features 🔧
- **FastAPI Backend**: High-performance async API
- **RESTful Endpoints**: Clean, documented API
- **File Handling**: Robust upload and processing
- **Error Management**: Comprehensive error handling

## 🚀 Quick Start

### One-Command Startup ⚡
```bash
./start.sh
```

This script will:
- ✅ Check prerequisites (Node.js, Python)
- 📦 Install dependencies automatically
- 🔧 Start backend server (port 8000)
- 🎨 Start frontend server (port 5173)
- 🌐 Open your browser to the application

### Manual Setup

#### Backend Setup
```bash
cd server
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

#### Frontend Setup
```bash
cd web
npm install
npm run dev
```

## 📱 Application URLs

- **Web Interface**: http://localhost:5173
- **API Documentation**: http://localhost:8000/docs
- **Backend API**: http://localhost:8000

### CLI Usage (Legacy Commands)

#### Single Image Verification
```bash
# Verify if an image contains enrolled person
python src/verify.py --image path/to/image.jpg
```

#### Multi-Face Detection
```bash
# Detect all faces in group photos
python src/multi_face_evaluator.py --image path/to/group_photo.jpg
```

#### Batch Evaluation
```bash
# Evaluate entire folder
python src/evaluate_queries.py --out_dir results/
python src/multi_face_evaluator.py --folder path/to/images/ --out_dir results/
```

## 📁 Project Structure

```
face_verification_project/
│
├── 🗂️ data/
│   ├── raw/                        # Original images
│   │   └── query_images/          # Test images
│   │       ├── single_face/       # PersonA reference images (30 files)
│   │       ├── multiple_faces/    # Group photos for testing
│   │       └── reference/         # Additional test images
│   ├── processed/                 # Preprocessed images (160x160)
│   │   ├── personA/              # PersonA processed images
│   │   └── query_images/         # Test dataset (514 images)
│   │       ├── positive/         # PersonA samples (14 images)
│   │       └── negative/         # Non-PersonA samples (500 images)
│   ├── embeddings/               # PersonA gallery embeddings
│   │   ├── personA_normalized.npz # Gallery vectors + metadata
│   │   └── personA_meta.json     # Embedding metadata
│   └── evaluation/               # Evaluation results
│       └── runs/                 # Timestamped evaluation runs
│
├── 🔧 src/                        # Source code
│   ├── preprocessing.py          # Image preprocessing pipeline
│   ├── inference.py             # Core verification engine
│   ├── generate_embeddings.py   # Gallery generation
│   ├── verify.py               # CLI verification tool
│   ├── evaluate_queries.py     # Standard evaluation
│   ├── extend_evaluation.py    # Extended evaluation system
│   ├── multi_face_evaluator.py # Multi-face detection
│   └── threshold_optimization.py # Threshold tuning
│
├── ⚙️ config/
│   └── threshold.json           # Optimized threshold configuration
│
├── 🧪 github/                   # Additional resources
│   ├── face_verification_copilot_script.txt
│   ├── requirement.txt
│   └── timeline.txt
│
├── test_multi_face.py           # Comprehensive test suite
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

## 🎯 Core Modules

### 1. Single Face Verification (`src/verify.py`)
- **Purpose**: Verify if an image contains PersonA
- **Usage**: `python src/verify.py --image path/to/image.jpg`
- **Output**: JSON verification report with similarity score and prediction

### 2. Multi-Face Detection (`src/multi_face_evaluator.py`)
- **Purpose**: Detect and verify multiple faces in group photos
- **Usage**: `python src/multi_face_evaluator.py --image path/to/group.jpg`
- **Features**: 
  - Detects all faces with bounding boxes
  - Verifies each face against PersonA gallery
  - Generates annotated visualizations
  - Batch processing capability

### 3. Evaluation System (`src/evaluate_queries.py`, `src/extend_evaluation.py`)
- **Purpose**: Comprehensive evaluation with metrics and visualizations
- **Usage**: `python src/extend_evaluation.py`
- **Features**:
  - ROC curves and performance metrics
  - Confusion matrix analysis
  - Similarity distribution plots
  - Reproducible evaluation runs

### 4. Preprocessing Pipeline (`src/preprocessing.py`)
- **Purpose**: Face detection, alignment, and standardization
- **Usage**: `python src/preprocessing.py --input_dir raw/ --output_dir processed/`
- **Features**: MTCNN detection, 160x160 alignment, batch processing

### 5. Gallery Management (`src/generate_embeddings.py`)
- **Purpose**: Generate and manage PersonA reference embeddings
- **Output**: Normalized embeddings with metadata for fast matching

## 📊 Performance Metrics

### Current System Performance (Nov 2025)
- **🎯 Accuracy**: 100% on 514-image evaluation dataset
- **⚡ Speed**: ~3.5 faces/second processing
- **🔍 Detection**: MTCNN with 0.6 confidence threshold  
- **🎚️ Verification**: Optimized threshold 0.6572 (Brute Force F1)
- **💾 Gallery**: 30 PersonA reference embeddings
- **📏 False Positive Rate**: 0.00% (perfect precision)

### Evaluation Results
```
Dataset Composition:
├── 14 Positive samples (PersonA from single_face)
├── 500 Negative samples (randomly sampled from others)
└── 514 Total evaluation images

Performance Matrix:
├── True Positives: 14/14 (100%)
├── True Negatives: 500/500 (100%)  
├── False Positives: 0/500 (0%)
└── False Negatives: 0/14 (0%)
```

## �️ Advanced Usage

### Configuration Management
```python
# Load optimized threshold
from src.inference import load_threshold
threshold = load_threshold("config/threshold.json")  # 0.6572
```

### API Usage
```python
from src.inference import FaceVerifier
from src.multi_face_evaluator import MultiFaceEvaluator

# Single face verification
verifier = FaceVerifier()
is_match, similarity = verifier.verify_faces("image1.jpg", "image2.jpg")

# Multi-face detection
evaluator = MultiFaceEvaluator()
result = evaluator.evaluate_image_multi("group_photo.jpg")
print(f"Found {result['detected_count']} faces")
```

### Batch Processing
```bash
# Process entire folder with annotations
python src/multi_face_evaluator.py \
  --folder data/raw/query_images/multiple_faces/ \
  --out_dir results/batch_run/ \
  --recursive

# Extended evaluation with visualization
python src/extend_evaluation.py
```

## 🔧 System Architecture

### Technology Stack
- **🧠 Models**: FaceNet (InceptionResnetV1), MTCNN
- **⚡ Acceleration**: Apple Silicon MPS, CUDA support
- **📊 ML Stack**: PyTorch, scikit-learn, NumPy
- **🖼️ Image Processing**: PIL, OpenCV, facenet-pytorch
- **📈 Visualization**: Matplotlib, seaborn
- **🏗️ Framework**: Modular Python architecture

### Key Components
1. **FaceVerifier**: Core verification engine with threshold management
2. **MultiFaceEvaluator**: Multi-face detection and batch processing
3. **Preprocessing Pipeline**: MTCNN-based face extraction and alignment
4. **Gallery System**: PersonA embedding storage and management
5. **Evaluation Framework**: Metrics, visualization, and reproducible runs
6. **CLI Tools**: Production-ready command-line interfaces

## 📈 Development Timeline

### ✅ **Completed (Nov 2025)**
- [x] **DAY 1-3**: Core infrastructure and preprocessing pipeline
- [x] **DAY 4-5**: Single-face verification with baseline threshold
- [x] **DAY 6-7**: Threshold optimization achieving 100% accuracy
- [x] **DAY 8-9**: Production pipeline with configuration management
- [x] **DAY 10-11**: Extended evaluation system with large-scale testing
- [x] **DAY 12-13**: Multi-face detection and visualization
- [x] **DAY 14**: Comprehensive testing and production readiness

### 🔬 **Key Milestones**
1. **Baseline Implementation**: FaceNet + MTCNN integration
2. **Threshold Optimization**: Brute force search achieving perfect F1
3. **Scalable Evaluation**: 25x dataset expansion (40→514 images)
4. **Multi-Face Capability**: Group photo processing with annotations
5. **Production Ready**: CLI tools + batch processing + error handling
## 📚 Documentation & Testing

### Testing
```bash
# Run comprehensive test suite
python test_multi_face.py

# Self-test individual modules
python src/multi_face_evaluator.py  # Multi-face self-test
python src/verify.py --help         # CLI documentation
```

### Output Formats
- **JSON Results**: Structured data with timestamps and metadata
- **Annotated Images**: Bounding boxes with similarity scores
- **CSV Reports**: Evaluation metrics and confusion matrices
- **Visualizations**: ROC curves and similarity distributions

## 🔧 Configuration

### Threshold Configuration (`config/threshold.json`)
```json
{
  "personA_threshold": 0.6572,
  "method": "brute_force_f1",
  "optimization_date": "2025-11-26",
  "accuracy": 1.0
}
```

### Environment Variables
```bash
# Optional: Set device preference
export PYTORCH_DEVICE=mps  # or cuda, cpu
```

## � Deployment

### Production Checklist
- ✅ **100% Accuracy**: Verified on 514-image evaluation set
- ✅ **Error Handling**: Robust processing with fallbacks
- ✅ **Scalability**: Batch processing for large datasets
- ✅ **Configuration**: Externalized threshold management
- ✅ **Monitoring**: Performance metrics and logging
- ✅ **CLI Tools**: Production-ready interfaces
- ✅ **Documentation**: Comprehensive usage examples

### Performance Optimization
- **Apple Silicon**: MPS acceleration for Mac M2
- **Batch Processing**: Efficient multi-image processing
- **Memory Management**: Optimized for large-scale evaluation
- **Caching**: Gallery embeddings loaded once

## 🤝 API Reference

### Core Classes
```python
# FaceVerifier: Single-face verification
from src.inference import FaceVerifier
verifier = FaceVerifier(verification_threshold=0.6572)

# MultiFaceEvaluator: Multi-face detection
from src.multi_face_evaluator import MultiFaceEvaluator
evaluator = MultiFaceEvaluator(emb_path="data/embeddings/personA_normalized.npz")
```

### CLI Commands

## 🏗️ System Architecture

```
face_verification_project/
├── server/                 # FastAPI backend
│   ├── main.py            # API server
│   ├── static/            # Static files
│   ├── uploads/           # Uploaded images
│   └── annotations/       # Processed results
├── web/                   # React frontend
│   ├── src/
│   │   ├── components/    # React components
│   │   ├── services/      # API services
│   │   └── utils/         # Utilities
│   ├── public/            # Static assets
│   └── package.json       # Dependencies
├── src/                   # Core ML pipeline
│   ├── inference.py       # Face verification
│   ├── multi_face_evaluator.py # Multi-face detection
│   ├── embedding_generator.py # Feature extraction
│   └── utils.py          # Helper functions
├── models/                # ML models
├── data/                  # Datasets and embeddings
├── start.sh              # Development startup
└── README.md             # This file
```

## 🔧 Technology Stack

### Backend
- **FastAPI**: Modern Python web framework
- **OpenCV**: Computer vision processing
- **NumPy**: Numerical computations
- **Pillow**: Image processing
- **face_recognition**: Face detection and encoding
- **Uvicorn**: ASGI server

### Frontend
- **React 18**: Modern React with hooks
- **Vite**: Fast build tool and dev server
- **Tailwind CSS**: Utility-first CSS framework
- **React Router**: Client-side routing
- **Axios**: HTTP client
- **Lucide React**: Beautiful icons

### Machine Learning
- **Face Recognition**: Deep learning embeddings
- **Multi-Face Detection**: Advanced face detection
- **Similarity Matching**: Cosine similarity with thresholds
- **Apple Silicon**: MPS acceleration support

## 📖 API Documentation

### REST Endpoints

#### 1. Enroll Person
```http
POST /api/enroll
Content-Type: multipart/form-data

person_name: string
file: image file
```

**Response:**
```json
{
  "status": "success",
  "message": "Person enrolled successfully",
  "person_name": "John Doe",
  "faces_count": 1
}
```

#### 2. Verify Faces
```http
POST /api/verify
Content-Type: multipart/form-data

file: image file
```

**Response:**
```json
{
  "total_faces": 2,
  "verified_faces": [
    {
      "person_name": "John Doe",
      "confidence": 0.85,
      "is_verified": true,
      "bbox": [100, 150, 200, 250]
    }
  ],
  "annotated_image_path": "/static/annotations/result.jpg",
  "processing_time": 1.23
}
```

#### 3. Threshold Management
```http
GET /api/threshold
POST /api/threshold
Content-Type: application/json

{
  "threshold": 0.5
}
```

## 🎯 Usage Guide

### 1. Enrolling a Person
1. Navigate to the "Enroll Person" tab
2. Enter the person's name
3. Upload a clear photo of their face
4. Click "Enroll Person"
5. System will detect and store the face

### 2. Verifying Faces
1. Go to the "Verify Faces" tab
2. Upload an image (can contain multiple faces)
3. Click "Verify Faces"
4. View results with confidence scores
5. Download annotated results

### 3. Configuring Settings
1. Visit the "Settings" tab
2. Adjust verification threshold (0.1 - 0.9)
3. Higher values = stricter verification
4. Lower values = more lenient verification
5. Save changes

## 🔍 How It Works

### Face Enrollment Process
1. **Image Upload**: User uploads person's photo
2. **Face Detection**: System detects faces in image
3. **Feature Extraction**: Generates face embeddings
4. **Database Storage**: Stores embeddings with person name
5. **Validation**: Confirms successful enrollment

### Face Verification Process
1. **Image Analysis**: Detect all faces in uploaded image
2. **Feature Extraction**: Generate embeddings for each face
3. **Similarity Matching**: Compare with enrolled faces
4. **Threshold Checking**: Apply configurable threshold
5. **Result Annotation**: Draw bounding boxes and labels
6. **Response Generation**: Return detailed results

### Threshold System
- **0.1-0.3**: Very permissive (may accept false positives)
- **0.4-0.6**: Balanced approach (recommended)
- **0.7-0.9**: Very strict (may reject valid faces)

## ⚙️ Configuration

### Environment Variables
- `VERIFICATION_THRESHOLD`: Default verification threshold (0.5)
- `MAX_UPLOAD_SIZE`: Maximum file size in bytes (10MB)
- `API_HOST`: API server host (0.0.0.0)
- `API_PORT`: API server port (8000)

### File Structure
- **uploads/**: Original uploaded images
- **static/annotations/**: Processed images with annotations
- **data/embeddings/**: Face recognition embeddings
- **models/**: Face recognition models

## 🚀 Deployment

### Docker (Recommended)
```bash
# Build and run with Docker Compose
docker-compose up --build
```

### Manual Deployment
1. **Backend**: Deploy FastAPI with uvicorn
2. **Frontend**: Build React app and serve static files
3. **Models**: Ensure face recognition models are available
4. **Storage**: Configure persistent storage for uploads

## 🐛 Troubleshooting

### Common Issues

1. **Backend won't start**
   - Check Python version (3.8+)
   - Install required packages: `pip install -r requirements.txt`
   - Verify face_recognition installation

2. **Frontend build fails**
   - Check Node.js version (16+)
   - Clear node_modules: `rm -rf node_modules && npm install`
   - Verify package.json dependencies

3. **Face detection not working**
   - Ensure good image quality and lighting
   - Check that faces are clearly visible
   - Try different image formats (JPG, PNG)

4. **API connection errors**
   - Verify backend is running on port 8000
   - Check CORS configuration in main.py
   - Test with curl: `curl http://localhost:8000/docs`

### Performance Tips
- Use JPG format for smaller file sizes
- Ensure good lighting in photos
- Crop images to focus on faces
- Use consistent image quality across enrollments

## 📋 Requirements

### System Requirements
- **Python**: 3.8 or higher
- **Node.js**: 16 or higher
- **Memory**: 4GB RAM minimum (8GB recommended)
- **Storage**: 2GB for models and uploads
- **OS**: Linux, macOS, or Windows

### Python Dependencies
```
fastapi>=0.104.1
uvicorn[standard]>=0.24.0
python-multipart>=0.0.6
opencv-python>=4.8.0
numpy>=1.24.0
Pillow>=10.0.0
face-recognition>=1.3.0
```

### Node.js Dependencies
```
react>=18.2.0
vite>=4.4.0
tailwindcss>=3.3.0
react-router-dom>=6.8.0
axios>=1.6.0
lucide-react>=0.263.0
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### Development Guidelines
- Follow PEP 8 for Python code
- Use React best practices and hooks
- Write clear, descriptive commit messages
- Add documentation for new features
- Test thoroughly before submitting

## 📝 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- OpenCV community for computer vision tools
- face_recognition library for robust face detection
- React team for the excellent framework
- FastAPI for the modern, fast API framework
- Tailwind CSS for beautiful, responsive design

---

**Built with ❤️ for secure and accurate face verification**
```bash
# Verify single image
python src/verify.py --image path/to/image.jpg

# Multi-face detection
python src/multi_face_evaluator.py --image path/to/group.jpg --out_dir results/

# Batch evaluation
python src/extend_evaluation.py

# Preprocessing
python src/preprocessing.py --input_dir raw/ --output_dir processed/
```

## 🆘 Troubleshooting

### Common Issues
1. **No face detected**: Check image quality and lighting
2. **MPS errors**: Fallback to CPU for MTCNN compatibility
3. **Memory issues**: Process images in smaller batches
4. **Threshold sensitivity**: Use provided optimized value (0.6572)

### Support Files
- **Timeline**: `github/timeline.txt` - Development progress
- **Requirements**: `github/requirement.txt` - Dependency management
- **Test Scripts**: `test_multi_face.py` - Validation suite

## 📄 License

This project is for educational and research purposes.

---
**� Face Verification System v1.0 - Production Ready**  
*Achieving 100% accuracy with optimized threshold and multi-face capabilities*