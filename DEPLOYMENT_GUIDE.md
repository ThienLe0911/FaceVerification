# DEPLOYMENT GUIDE
# Hướng dẫn triển khai hệ thống Face Verification

## 1. System Requirements

### 1.1 Hardware Requirements
```
Minimum:
- CPU: 4 cores, 2.0GHz
- RAM: 8GB
- Storage: 5GB free space
- Network: 100Mbps

Recommended:
- CPU: 8 cores, 3.0GHz or higher
- RAM: 16GB or higher  
- GPU: NVIDIA with CUDA (optional, for acceleration)
- Storage: 20GB free space (SSD preferred)
- Network: 1Gbps
```

### 1.2 Software Dependencies
```bash
# Core Requirements
Python 3.8+
Node.js 16+
npm 8+

# Optional (for GPU acceleration)
CUDA Toolkit 11.7+
cuDNN 8.5+
```

## 2. Installation Steps

### 2.1 Backend Setup
```bash
# 1. Clone repository
cd /path/to/your/workspace
git clone <your-repo> face_verification_project
cd face_verification_project

# 2. Create Python virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate    # Windows

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Download pre-trained models (if not included)
python -c "
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
mtcnn = MTCNN()
resnet = InceptionResnetV1(pretrained='vggface2').eval()
print('Models downloaded successfully')
"

# 5. Test backend
cd server
python simple_server.py
# Should see: Uvicorn running on http://0.0.0.0:8001
```

### 2.2 Frontend Setup
```bash
# 1. Navigate to web directory
cd ../web

# 2. Install Node.js dependencies
npm install

# 3. Configure environment
cat > .env.local << EOF
VITE_API_URL=http://localhost:8001
VITE_DEMO_MODE=false
EOF

# 4. Build production version
npm run build

# 5. Test frontend (development)
npm run dev
# Access: http://localhost:5173
```

### 2.3 Production Configuration
```bash
# 1. Install production web server
npm install -g serve

# 2. Serve built frontend
serve -s dist -l 3000
# Frontend: http://localhost:3000

# 3. Run backend with production settings
cd ../server
uvicorn simple_server:app --host 0.0.0.0 --port 8001 --workers 4
# Backend API: http://localhost:8001
```

## 3. Configuration Options

### 3.1 Backend Configuration (`server/config.py`)
```python
# Face Detection Settings
MTCNN_CONF_THRESHOLD = 0.9      # Minimum face detection confidence
MIN_FACE_SIZE = 30              # Minimum face size in pixels
MAX_FACES_PER_IMAGE = 3         # Maximum faces to detect

# Quality Scoring
MIN_AREA_RATIO = 0.003          # 0.3% minimum face area
QUALITY_THRESHOLD = 60          # Minimum quality score

# Performance Settings
MAX_IMAGE_SIZE = 2048           # Max image dimension (pixels)
REQUEST_TIMEOUT = 30            # API request timeout (seconds)
MAX_CONCURRENT_REQUESTS = 10    # Maximum simultaneous requests

# Storage Settings
UPLOAD_DIR = "./uploads"        # Uploaded images directory
FACE_DB_DIR = "./face_database" # Face embeddings storage
LOG_LEVEL = "INFO"              # Logging level
```

### 3.2 Frontend Configuration (`web/.env`)
```bash
# API Configuration
VITE_API_URL=http://your-backend-server:8001
VITE_API_TIMEOUT=30000

# Feature Flags
VITE_DEMO_MODE=false           # Set true for demo without real API
VITE_ENABLE_DEBUG=false        # Show debug information
VITE_ENABLE_BATCH_UPLOAD=true  # Enable multiple file upload
VITE_MAX_FILE_SIZE=5242880     # 5MB max file size

# UI Configuration
VITE_THEME=default
VITE_LANGUAGE=vi               # vi | en
VITE_SHOW_CONFIDENCE=true      # Show detection confidence
```

## 4. Docker Deployment

### 4.1 Create Dockerfile (Backend)
```dockerfile
# server/Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8001

# Run application
CMD ["uvicorn", "simple_server:app", "--host", "0.0.0.0", "--port", "8001"]
```

### 4.2 Create Dockerfile (Frontend)
```dockerfile
# web/Dockerfile
FROM node:16-alpine as builder

WORKDIR /app

# Copy package files
COPY package*.json ./
RUN npm ci

# Copy source code and build
COPY . .
RUN npm run build

# Production stage
FROM nginx:alpine

# Copy built files
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy nginx configuration
COPY nginx.conf /etc/nginx/nginx.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
```

### 4.3 Docker Compose
```yaml
# docker-compose.yml
version: '3.8'

services:
  backend:
    build: 
      context: ./server
      dockerfile: Dockerfile
    ports:
      - "8001:8001"
    volumes:
      - ./uploads:/app/uploads
      - ./face_database:/app/face_database
    environment:
      - PYTHONUNBUFFERED=1
      - LOG_LEVEL=INFO
    restart: unless-stopped

  frontend:
    build:
      context: ./web
      dockerfile: Dockerfile
    ports:
      - "80:80"
    depends_on:
      - backend
    restart: unless-stopped
    environment:
      - VITE_API_URL=http://backend:8001

volumes:
  uploads:
  face_database:
```

### 4.4 Deploy with Docker
```bash
# Build and run
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f backend
docker-compose logs -f frontend

# Stop services
docker-compose down
```

## 5. Production Deployment

### 5.1 NGINX Configuration
```nginx
# /etc/nginx/sites-available/face-verification
server {
    listen 80;
    server_name your-domain.com;

    # Frontend
    location / {
        proxy_pass http://localhost:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }

    # Backend API
    location /api/ {
        proxy_pass http://localhost:8001;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Increase timeout for face processing
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
        
        # Increase max body size for image uploads
        client_max_body_size 10M;
    }
}

# Enable HTTPS (recommended)
server {
    listen 443 ssl http2;
    server_name your-domain.com;

    ssl_certificate /path/to/ssl/cert.pem;
    ssl_certificate_key /path/to/ssl/key.pem;

    # Same configuration as above...
}
```

### 5.2 Systemd Services

#### Backend Service
```ini
# /etc/systemd/system/face-verification-backend.service
[Unit]
Description=Face Verification Backend
After=network.target

[Service]
Type=exec
User=www-data
Group=www-data
WorkingDirectory=/path/to/face_verification_project/server
Environment=PATH=/path/to/face_verification_project/venv/bin
ExecStart=/path/to/face_verification_project/venv/bin/uvicorn simple_server:app --host 0.0.0.0 --port 8001 --workers 4
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

#### Frontend Service
```ini
# /etc/systemd/system/face-verification-frontend.service
[Unit]
Description=Face Verification Frontend
After=network.target

[Service]
Type=exec
User=www-data
Group=www-data
WorkingDirectory=/path/to/face_verification_project/web
ExecStart=/usr/bin/serve -s dist -l 3000
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

#### Enable and Start Services
```bash
# Enable services
sudo systemctl enable face-verification-backend
sudo systemctl enable face-verification-frontend

# Start services
sudo systemctl start face-verification-backend
sudo systemctl start face-verification-frontend

# Check status
sudo systemctl status face-verification-backend
sudo systemctl status face-verification-frontend
```

## 6. Monitoring and Logging

### 6.1 Log Configuration
```python
# server/logging_config.py
import logging
import logging.handlers

def setup_logging():
    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            # File handler with rotation
            logging.handlers.RotatingFileHandler(
                'logs/face_verification.log',
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5
            ),
            # Console handler
            logging.StreamHandler()
        ]
    )
```

### 6.2 Health Check Endpoint
```python
# Add to server/simple_server.py
@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    try:
        # Test MTCNN loading
        from facenet_pytorch import MTCNN
        mtcnn = MTCNN()
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0",
            "components": {
                "mtcnn": "loaded",
                "facenet": "loaded"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")
```

### 6.3 Monitoring Script
```bash
#!/bin/bash
# scripts/monitor.sh

# Check backend health
curl -f http://localhost:8001/health > /dev/null
if [ $? -eq 0 ]; then
    echo "✅ Backend healthy"
else
    echo "❌ Backend unhealthy"
    # Restart service
    sudo systemctl restart face-verification-backend
fi

# Check frontend
curl -f http://localhost:3000 > /dev/null
if [ $? -eq 0 ]; then
    echo "✅ Frontend healthy"
else
    echo "❌ Frontend unhealthy"
    sudo systemctl restart face-verification-frontend
fi

# Check disk space
DISK_USAGE=$(df /path/to/face_verification_project | tail -1 | awk '{print $5}' | sed 's/%//')
if [ $DISK_USAGE -gt 80 ]; then
    echo "⚠️ Disk usage high: ${DISK_USAGE}%"
fi
```

## 7. Backup and Recovery

### 7.1 Backup Script
```bash
#!/bin/bash
# scripts/backup.sh

BACKUP_DIR="/backup/face_verification/$(date +%Y%m%d_%H%M%S)"
PROJECT_DIR="/path/to/face_verification_project"

mkdir -p $BACKUP_DIR

# Backup face database
tar -czf $BACKUP_DIR/face_database.tar.gz -C $PROJECT_DIR face_database/

# Backup configuration
cp $PROJECT_DIR/server/config.py $BACKUP_DIR/
cp $PROJECT_DIR/web/.env.local $BACKUP_DIR/

# Backup logs (last 7 days)
find $PROJECT_DIR/logs -name "*.log" -mtime -7 -exec cp {} $BACKUP_DIR/ \;

echo "Backup completed: $BACKUP_DIR"
```

### 7.2 Recovery Procedure
```bash
# 1. Stop services
sudo systemctl stop face-verification-backend
sudo systemctl stop face-verification-frontend

# 2. Restore face database
tar -xzf backup/face_database.tar.gz -C /path/to/face_verification_project/

# 3. Restore configuration
cp backup/config.py /path/to/face_verification_project/server/
cp backup/.env.local /path/to/face_verification_project/web/

# 4. Restart services
sudo systemctl start face-verification-backend
sudo systemctl start face-verification-frontend
```

## 8. Security Considerations

### 8.1 API Security
```python
# Add to server/simple_server.py
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-domain.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Trusted host middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["your-domain.com", "*.your-domain.com"]
)
```

### 8.2 File Security
```python
# Secure file handling
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

def validate_file(file):
    # Check file extension
    if not any(file.filename.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS):
        raise HTTPException(400, "Invalid file type")
    
    # Check file size
    file.seek(0, 2)  # Seek to end
    size = file.tell()
    file.seek(0)     # Reset position
    
    if size > MAX_FILE_SIZE:
        raise HTTPException(400, "File too large")
```

### 8.3 Data Privacy
```bash
# Set appropriate file permissions
chmod 700 face_database/
chmod 600 face_database/*

# Encrypt sensitive data at rest
# Use tools like gpg or dm-crypt for encryption
```

## 9. Performance Optimization

### 9.1 GPU Acceleration
```python
# Enable GPU for MTCNN and FaceNet
import torch

# Check GPU availability
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device('cpu')
    print("Using CPU")

# Initialize models with GPU
mtcnn = MTCNN(device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
```

### 9.2 Caching Strategy
```python
# Add Redis for caching
import redis

redis_client = redis.Redis(host='localhost', port=6379, db=0)

def cache_face_embedding(person_id, embedding):
    """Cache face embedding for faster verification."""
    key = f"embedding:{person_id}"
    redis_client.set(key, embedding.tobytes(), ex=3600)  # 1 hour cache

def get_cached_embedding(person_id):
    """Retrieve cached embedding."""
    key = f"embedding:{person_id}"
    data = redis_client.get(key)
    if data:
        return np.frombuffer(data, dtype=np.float32)
    return None
```

## 10. Troubleshooting

### 10.1 Common Issues

#### Backend Won't Start
```bash
# Check Python version
python --version

# Check dependencies
pip list | grep torch
pip list | grep facenet

# Check ports
sudo netstat -tulpn | grep 8001

# Check logs
tail -f logs/face_verification.log
```

#### Frontend Build Errors
```bash
# Clear cache
npm cache clean --force
rm -rf node_modules package-lock.json
npm install

# Check Node version
node --version
npm --version
```

#### Face Detection Errors
```bash
# Test MTCNN installation
python -c "
from facenet_pytorch import MTCNN
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
mtcnn = MTCNN()
print('MTCNN loaded successfully')
"
```

### 10.2 Performance Issues
```bash
# Monitor system resources
htop
nvidia-smi  # If using GPU

# Profile application
python -m cProfile server/simple_server.py

# Check network latency
ping your-domain.com
```

This deployment guide provides comprehensive instructions for setting up the Face Verification system in production environments with proper security, monitoring, and maintenance procedures.