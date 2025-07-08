#!/usr/bin/env python3
"""
Test script to verify all imports work correctly
"""

import os
import sys

# Disable xFormers to avoid issues
os.environ["XFORMERS_DISABLED"] = "1"

def test_pytorch():
    """Test PyTorch and CUDA"""
    print("🔧 Testing PyTorch...")
    try:
        import torch
        print(f"  ✓ PyTorch version: {torch.__version__}")
        print(f"  ✓ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  ✓ GPU: {torch.cuda.get_device_name(0)}")
            print(f"  ✓ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        return True
    except Exception as e:
        print(f"  ✗ PyTorch error: {e}")
        return False

def test_diffusers():
    """Test Diffusers import"""
    print("\n📦 Testing Diffusers...")
    try:
        from diffusers import DiffusionPipeline
        print("  ✓ Diffusers import successful")
        return True
    except Exception as e:
        print(f"  ✗ Diffusers error: {e}")
        return False

def test_app_imports():
    """Test app imports"""
    print("\n🏗️ Testing App Imports...")
    try:
        sys.path.insert(0, '.')
        from app.models import VideoGenerationRequest
        print("  ✓ App models import successful")
        
        from app.auth import authenticate_user
        print("  ✓ App auth import successful")
        
        from app.animatediff_handler import AnimateDiffHandler
        print("  ✓ AnimateDiff handler import successful")
        
        return True
    except Exception as e:
        print(f"  ✗ App import error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_fastapi():
    """Test FastAPI import"""
    print("\n🌐 Testing FastAPI...")
    try:
        import fastapi
        import uvicorn
        print("  ✓ FastAPI imports successful")
        return True
    except Exception as e:
        print(f"  ✗ FastAPI error: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 50)
    print("🧪 AnimateDiff Import Test")
    print("=" * 50)
    
    results = []
    results.append(test_pytorch())
    results.append(test_diffusers())
    results.append(test_fastapi())
    results.append(test_app_imports())
    
    print("\n" + "=" * 50)
    print("📊 Test Results")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"🎉 All tests passed! ({passed}/{total})")
        print("✅ Your environment is ready!")
        print("\nYou can now run: deploy\\windows\\run_dev.bat")
    else:
        print(f"⚠️  Some tests failed ({passed}/{total})")
        print("❌ Please fix the errors above before running the server")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)