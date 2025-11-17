# test_mps_detailed.py
import torch
import platform
import time

def test_mps_functionality():
    """Test chi tiết khả năng sử dụng MPS"""
    
    print("=" * 60)
    print("🍎 KIỂM TRA APPLE SILICON MPS CHI TIẾT")
    print("=" * 60)
    
    # Thông tin hệ thống
    print(f"🖥️  Hệ thống: {platform.system()} {platform.release()}")
    print(f"⚙️  Kiến trúc: {platform.machine()}")
    print(f"🐍 Python: {platform.python_version()}")
    print(f"🔥 PyTorch: {torch.__version__}")
    
    print("\n" + "=" * 40)
    print("MPS STATUS:")
    print("=" * 40)
    
    # Kiểm tra MPS
    try:
        mps_available = torch.backends.mps.is_available()
        mps_built = torch.backends.mps.is_built()
        
        print(f"✅ MPS Available: {mps_available}")
        print(f"✅ MPS Built: {mps_built}")
        
        if mps_available and mps_built:
            print("🎉 Apple Silicon GPU sẵn sàng sử dung!")
            
            # Test tạo tensor trên MPS
            print("\n🧪 TESTING MPS FUNCTIONALITY:")
            print("-" * 40)
            
            try:
                # Tạo tensor trên CPU
                cpu_tensor = torch.randn(1000, 1000)
                print(f"✅ CPU tensor tạo thành công: {cpu_tensor.device}")
                
                # Chuyển lên MPS
                mps_tensor = cpu_tensor.to('mps')
                print(f"✅ MPS tensor tạo thành công: {mps_tensor.device}")
                
                # Test phép toán trên MPS
                start_time = time.time()
                result_mps = torch.mm(mps_tensor, mps_tensor.T)
                mps_time = time.time() - start_time
                print(f"✅ Matrix multiplication trên MPS: {mps_time:.4f}s")
                
                # So sánh với CPU
                start_time = time.time()
                result_cpu = torch.mm(cpu_tensor, cpu_tensor.T)
                cpu_time = time.time() - start_time
                print(f"⏰ Matrix multiplication trên CPU: {cpu_time:.4f}s")
                
                if mps_time < cpu_time:
                    speedup = cpu_time / mps_time
                    print(f"🚀 MPS nhanh hơn CPU {speedup:.2f}x!")
                else:
                    print("💡 CPU nhanh hơn (có thể do tensor nhỏ)")
                
                # Kiểm tra memory
                if hasattr(torch.mps, 'current_allocated_memory'):
                    memory = torch.mps.current_allocated_memory()
                    print(f"🧠 MPS memory đang sử dụng: {memory / 1024 / 1024:.2f} MB")
                
                print("\n✅ Tất cả test MPS THÀNH CÔNG!")
                return True
                
            except Exception as e:
                print(f"❌ Lỗi khi test MPS functionality: {e}")
                return False
        else:
            print("❌ MPS không khả dụng trên hệ thống này")
            return False
            
    except Exception as e:
        print(f"❌ Lỗi khi kiểm tra MPS: {e}")
        return False

def recommend_device():
    """Đưa ra khuyến nghị device tốt nhất"""
    print("\n" + "=" * 40)
    print("🎯 KHUYẾN NGHỊ DEVICE:")
    print("=" * 40)
    
    if torch.cuda.is_available():
        print("🚀 Nên sử dụng: 'cuda' (NVIDIA GPU)")
        return 'cuda'
    elif torch.backends.mps.is_available():
        print("🍎 Nên sử dụng: 'mps' (Apple Silicon GPU)")
        return 'mps'
    else:
        print("💻 Sử dụng: 'cpu' (CPU only)")
        return 'cpu'

if __name__ == "__main__":
    success = test_mps_functionality()
    recommended_device = recommend_device()
    
    print("\n" + "=" * 60)
    print("📝 TÓM TẮT:")
    print("=" * 60)
    print(f"✅ MPS Test: {'PASS' if success else 'FAIL'}")
    print(f"🎯 Device khuyến nghị: '{recommended_device}'")
    print("💡 Sử dụng device này trong FaceVerifier để có hiệu suất tốt nhất!")
    print("=" * 60)