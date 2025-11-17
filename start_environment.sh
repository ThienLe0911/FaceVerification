#!/bin/bash

# Script khởi động môi trường Face Verification Project
echo "🚀 Đang khởi động môi trường Face Verification..."

# Di chuyển vào thư mục dự án
cd "$(dirname "$0")"

# Kích hoạt virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "✅ Đã kích hoạt virtual environment"
else
    echo "❌ Không tìm thấy virtual environment. Hãy chạy setup script trước."
    exit 1
fi

# Kiểm tra Python environment
python -c "
import torch
print(f'🐍 Python environment: OK')
print(f'🤖 PyTorch version: {torch.__version__}')
if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print('🍎 Apple Silicon GPU: Có sẵn')
else:
    print('💻 Apple Silicon GPU: Không có (sử dụng CPU)')
"

echo ""
echo "🎯 Môi trường đã sẵn sàng!"
echo "📚 Các lệnh hữu ích:"
echo "  - jupyter notebook notebooks/           # Mở Jupyter notebooks"
echo "  - python src/preprocessing.py          # Test preprocessing"
echo "  - python src/inference.py             # Test inference"
echo ""
echo "📂 Cấu trúc dự án:"
echo "  - data/raw/        : Đặt ảnh gốc vào đây"
echo "  - data/processed/  : Ảnh đã xử lý"
echo "  - notebooks/       : Jupyter notebooks"
echo "  - src/            : Source code"
echo ""
echo "🚀 Bắt đầu với: jupyter notebook notebooks/"