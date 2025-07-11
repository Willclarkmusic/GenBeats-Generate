#!/usr/bin/env python3
"""
Test script for video generation (quick validation)
"""

import os
import sys
import asyncio
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Disable xFormers to avoid issues
os.environ["XFORMERS_DISABLED"] = "1"

async def test_generation_pipeline():
    """Test the generation pipeline without full model loading"""
    print("🎬 Testing Generation Pipeline...")
    
    try:
        sys.path.insert(0, '.')
        from app.models import VideoGenerationRequest, Job
        from app.queue_manager import queue_manager
        from app.animatediff_handler import animatediff_handler
        
        # Create a test request
        request = VideoGenerationRequest(
            prompt="A beautiful sunset over mountains",
            steps=1,  # Very low steps for quick test
            width=512,
            height=512,
            duration=8  # Short duration
        )
        
        print(f"  ✓ Test request created: {request.prompt}")
        
        # Add job to queue
        job_id = await queue_manager.add_job(request)
        print(f"  ✓ Job added to queue: {job_id}")
        
        # Get the job
        job = await queue_manager.get_job(job_id)
        print(f"  ✓ Job retrieved: {job.status}")
        
        # Check handler device
        print(f"  ✓ Handler device: {animatediff_handler.device}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test function"""
    print("=" * 60)
    print("🧪 AnimateDiff Generation Pipeline Test")
    print("=" * 60)
    
    success = await test_generation_pipeline()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 Pipeline test passed!")
        print("✅ The async issues should be fixed!")
        print("\nYou can now try generating a video through the web interface.")
    else:
        print("❌ Pipeline test failed!")
        print("Please check the error messages above.")
    print("=" * 60)
    
    return success

if __name__ == "__main__":
    result = asyncio.run(main())
    sys.exit(0 if result else 1)