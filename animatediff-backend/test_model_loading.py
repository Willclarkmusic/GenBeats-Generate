#!/usr/bin/env python3
"""
Test script specifically for model loading
"""

import os
import sys
import asyncio
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Disable xFormers to avoid issues
os.environ["XFORMERS_DISABLED"] = "1"

async def test_model_loading():
    """Test AnimateDiff model loading"""
    print("🤖 Testing AnimateDiff Model Loading...")
    
    try:
        # Import after setting environment
        sys.path.insert(0, '.')
        from app.animatediff_handler import AnimateDiffHandler
        
        # Create handler
        handler = AnimateDiffHandler()
        print(f"  ✓ Handler created, device: {handler.device}")
        
        # Test model loading
        print("  📦 Loading models (this may take a few minutes)...")
        success = await handler.load_model()
        
        if success:
            print("  ✅ Model loading successful!")
            
            # Test model info
            info = handler.get_model_info()
            print(f"  📊 Model Info:")
            print(f"    - Model loaded: {info['model_loaded']}")
            print(f"    - Device: {info['device']}")
            print(f"    - GPU available: {info['gpu_available']}")
            if info['gpu_name']:
                print(f"    - GPU: {info['gpu_name']}")
            
            return True
        else:
            print("  ❌ Model loading failed!")
            return False
            
    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test function"""
    print("=" * 60)
    print("🧪 AnimateDiff Model Loading Test")
    print("=" * 60)
    
    success = await test_model_loading()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 Model loading test passed!")
        print("✅ Your environment is ready for video generation!")
        print("\nYou can now run: deploy\\windows\\run_dev.bat")
    else:
        print("❌ Model loading test failed!")
        print("Please check the error messages above.")
    print("=" * 60)
    
    return success

if __name__ == "__main__":
    result = asyncio.run(main())
    sys.exit(0 if result else 1)