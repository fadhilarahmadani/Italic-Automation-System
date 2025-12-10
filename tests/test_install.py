# tests/test_install.py
import sys

def test_libraries():
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        
        import transformers
        print(f"✅ Transformers: {transformers.__version__}")
        
        import datasets
        print(f"✅ Datasets: {datasets.__version__}")
        
        import sklearn
        print(f"✅ Scikit-learn: {sklearn.__version__}")
        
        import fastapi
        print(f"✅ FastAPI: {fastapi.__version__}")
        
        # Test CUDA availability
        cuda_available = torch.cuda.is_available()
        device = "CUDA" if cuda_available else "CPU"
        print(f"✅ Device: {device}")
        
        print("\n🎉 Semua library berhasil terinstal!")
        return True
        
    except ImportError as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    test_libraries()
