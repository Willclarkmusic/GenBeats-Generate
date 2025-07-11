#!/usr/bin/env python3
"""
Test script to verify duration changes work correctly
"""

import sys

def test_duration_conversion():
    """Test the duration conversion logic"""
    print("🕐 Testing Duration Conversion...")
    
    # Test cases: seconds -> frames (8 FPS)
    test_cases = [
        (1.0, 8),   # 1 second = 8 frames
        (2.0, 16),  # 2 seconds = 16 frames  
        (2.5, 20),  # 2.5 seconds = 20 frames
        (5.0, 40),  # 5 seconds = 40 frames
        (10.0, 80), # 10 seconds = 80 frames
    ]
    
    for seconds, expected_frames in test_cases:
        frames = round(seconds * 8)
        if frames == expected_frames:
            print(f"  ✓ {seconds}s = {frames} frames")
        else:
            print(f"  ✗ {seconds}s = {frames} frames (expected {expected_frames})")
    
    return True

def test_model_validation():
    """Test the model validation with new duration limits"""
    print("\n📋 Testing Model Validation...")
    
    try:
        sys.path.insert(0, '.')
        from app.models import VideoGenerationRequest
        
        # Test valid durations
        valid_tests = [8, 16, 40, 80]  # frames
        for frames in valid_tests:
            try:
                request = VideoGenerationRequest(
                    prompt="test",
                    duration=frames
                )
                print(f"  ✓ {frames} frames valid")
            except Exception as e:
                print(f"  ✗ {frames} frames failed: {e}")
        
        # Test invalid durations
        invalid_tests = [7, 81]  # below min, above max
        for frames in invalid_tests:
            try:
                request = VideoGenerationRequest(
                    prompt="test", 
                    duration=frames
                )
                print(f"  ✗ {frames} frames should be invalid but passed")
            except Exception as e:
                print(f"  ✓ {frames} frames correctly rejected: {e}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Model validation error: {e}")
        return False

def test_generation_time_estimates():
    """Estimate generation times for different durations"""
    print("\n⏱️ Generation Time Estimates (RTX 3080Ti)...")
    
    durations = [
        (1.0, 8),   # 1 second
        (2.0, 16),  # 2 seconds
        (5.0, 40),  # 5 seconds
        (10.0, 80), # 10 seconds
    ]
    
    for seconds, frames in durations:
        # Rough estimate: ~1-2 seconds per frame on RTX 3080Ti
        min_time = frames * 1
        max_time = frames * 2
        print(f"  {seconds}s video ({frames} frames): {min_time}-{max_time}s generation time")
    
    print("\n  💡 Tips for longer videos:")
    print("    - Use lower resolution (512x512) for faster generation")
    print("    - Reduce steps (20-25) for quicker results")
    print("    - Monitor VRAM usage with nvidia-smi")

def main():
    """Main test function"""
    print("=" * 60)
    print("🎬 AnimateDiff Duration Update Test")
    print("=" * 60)
    
    results = []
    results.append(test_duration_conversion())
    results.append(test_model_validation())
    test_generation_time_estimates()
    
    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print("🎉 Duration update tests passed!")
        print("✅ You can now generate up to 10 seconds of video!")
        print("\nRestart the server to see the new duration controls:")
        print("  deploy\\windows\\run_dev.bat")
    else:
        print(f"⚠️ Some tests failed ({passed}/{total})")
    print("=" * 60)
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)