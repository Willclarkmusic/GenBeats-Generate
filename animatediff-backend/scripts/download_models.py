#!/usr/bin/env python3
"""
Model download script for AnimateDiff backend.
This script downloads and caches all required models for AnimateDiff video generation.
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import List, Dict, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_environment():
    """Set up environment variables for model caching"""
    model_cache_dir = Path(os.getenv("MODEL_CACHE_DIR", "./models"))
    model_cache_dir.mkdir(exist_ok=True)
    
    # Set environment variables
    os.environ["TORCH_HOME"] = str(model_cache_dir)
    os.environ["HF_HOME"] = str(model_cache_dir)
    os.environ["TRANSFORMERS_CACHE"] = str(model_cache_dir)
    os.environ["DIFFUSERS_CACHE"] = str(model_cache_dir)
    
    logger.info(f"Model cache directory: {model_cache_dir}")
    return model_cache_dir

def check_gpu_availability():
    """Check if GPU is available"""
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"GPU available: {gpu_name}")
        else:
            logger.info("GPU not available, will use CPU")
        return gpu_available
    except ImportError:
        logger.warning("PyTorch not installed, cannot check GPU availability")
        return False

def download_base_model(model_id: str, cache_dir: Path) -> bool:
    """Download base diffusion model"""
    try:
        logger.info(f"Downloading base model: {model_id}")
        
        from diffusers import DiffusionPipeline
        
        # Download with low memory usage
        pipeline = DiffusionPipeline.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        logger.info(f"✓ Base model downloaded: {model_id}")
        return True
        
    except Exception as e:
        logger.error(f"✗ Failed to download base model {model_id}: {e}")
        return False

def download_motion_adapter(model_id: str, cache_dir: Path) -> bool:
    """Download AnimateDiff motion adapter"""
    try:
        logger.info(f"Downloading motion adapter: {model_id}")
        
        from diffusers import MotionAdapter
        
        adapter = MotionAdapter.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        logger.info(f"✓ Motion adapter downloaded: {model_id}")
        return True
        
    except Exception as e:
        logger.error(f"✗ Failed to download motion adapter {model_id}: {e}")
        return False

def download_animatediff_pipeline(base_model: str, motion_adapter: str, cache_dir: Path) -> bool:
    """Download complete AnimateDiff pipeline"""
    try:
        logger.info("Setting up AnimateDiff pipeline...")
        
        from diffusers import AnimateDiffPipeline, MotionAdapter, DPMSolverMultistepScheduler
        import torch
        
        # Load motion adapter
        adapter = MotionAdapter.from_pretrained(
            motion_adapter,
            cache_dir=cache_dir,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        # Load pipeline
        pipeline = AnimateDiffPipeline.from_pretrained(
            base_model,
            motion_adapter=adapter,
            cache_dir=cache_dir,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        # Set scheduler
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            pipeline.scheduler.config
        )
        
        logger.info("✓ AnimateDiff pipeline setup complete")
        return True
        
    except Exception as e:
        logger.error(f"✗ Failed to setup AnimateDiff pipeline: {e}")
        return False

def download_safety_checker(cache_dir: Path) -> bool:
    """Download safety checker model"""
    try:
        logger.info("Downloading safety checker...")
        
        from transformers import CLIPTextModel, CLIPTokenizer
        
        # Download CLIP text encoder
        tokenizer = CLIPTokenizer.from_pretrained(
            "openai/clip-vit-large-patch14",
            cache_dir=cache_dir
        )
        
        text_encoder = CLIPTextModel.from_pretrained(
            "openai/clip-vit-large-patch14",
            cache_dir=cache_dir
        )
        
        logger.info("✓ Safety checker downloaded")
        return True
        
    except Exception as e:
        logger.error(f"✗ Failed to download safety checker: {e}")
        return False

def verify_models(cache_dir: Path) -> Dict[str, bool]:
    """Verify that all models are properly downloaded"""
    logger.info("Verifying downloaded models...")
    
    results = {}
    
    try:
        # Test base model
        from diffusers import DiffusionPipeline
        pipeline = DiffusionPipeline.from_pretrained(
            "SG161222/Realistic_Vision_V4.0_noVAE",
            cache_dir=cache_dir,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        results["base_model"] = True
        logger.info("✓ Base model verification passed")
        
    except Exception as e:
        results["base_model"] = False
        logger.error(f"✗ Base model verification failed: {e}")
    
    try:
        # Test motion adapter
        from diffusers import MotionAdapter
        adapter = MotionAdapter.from_pretrained(
            "guoyww/animatediff-motion-adapter-v1-5-2",
            cache_dir=cache_dir,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        results["motion_adapter"] = True
        logger.info("✓ Motion adapter verification passed")
        
    except Exception as e:
        results["motion_adapter"] = False
        logger.error(f"✗ Motion adapter verification failed: {e}")
    
    try:
        # Test complete pipeline
        from diffusers import AnimateDiffPipeline, MotionAdapter
        
        adapter = MotionAdapter.from_pretrained(
            "guoyww/animatediff-motion-adapter-v1-5-2",
            cache_dir=cache_dir,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        pipeline = AnimateDiffPipeline.from_pretrained(
            "SG161222/Realistic_Vision_V4.0_noVAE",
            motion_adapter=adapter,
            cache_dir=cache_dir,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        results["complete_pipeline"] = True
        logger.info("✓ Complete pipeline verification passed")
        
    except Exception as e:
        results["complete_pipeline"] = False
        logger.error(f"✗ Complete pipeline verification failed: {e}")
    
    return results

def main():
    """Main function to download all models"""
    logger.info("Starting AnimateDiff model download...")
    
    try:
        # Import required packages
        import torch
        import diffusers
        import transformers
        
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"Diffusers version: {diffusers.__version__}")
        logger.info(f"Transformers version: {transformers.__version__}")
        
    except ImportError as e:
        logger.error(f"Required packages not installed: {e}")
        logger.error("Please install requirements first: pip install -r requirements.txt")
        return False
    
    # Setup environment
    cache_dir = setup_environment()
    gpu_available = check_gpu_availability()
    
    # Model configurations
    models_to_download = [
        {
            "name": "Base Model (Realistic Vision V4.0)",
            "id": "SG161222/Realistic_Vision_V4.0_noVAE",
            "type": "base"
        },
        {
            "name": "Motion Adapter (AnimateDiff V1.5)",
            "id": "guoyww/animatediff-motion-adapter-v1-5-2",
            "type": "motion_adapter"
        }
    ]
    
    success_count = 0
    total_count = len(models_to_download)
    
    # Download models
    for model_config in models_to_download:
        logger.info(f"Processing: {model_config['name']}")
        
        try:
            if model_config["type"] == "base":
                success = download_base_model(model_config["id"], cache_dir)
            elif model_config["type"] == "motion_adapter":
                success = download_motion_adapter(model_config["id"], cache_dir)
            else:
                logger.warning(f"Unknown model type: {model_config['type']}")
                continue
                
            if success:
                success_count += 1
                
        except Exception as e:
            logger.error(f"Error downloading {model_config['name']}: {e}")
    
    # Download safety checker
    logger.info("Downloading safety checker...")
    if download_safety_checker(cache_dir):
        success_count += 1
    total_count += 1
    
    # Setup complete pipeline
    logger.info("Setting up complete AnimateDiff pipeline...")
    if download_animatediff_pipeline(
        "SG161222/Realistic_Vision_V4.0_noVAE",
        "guoyww/animatediff-motion-adapter-v1-5-2",
        cache_dir
    ):
        success_count += 1
    total_count += 1
    
    # Verify models
    logger.info("Verifying all models...")
    verification_results = verify_models(cache_dir)
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("DOWNLOAD SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Successfully downloaded: {success_count}/{total_count} components")
    logger.info(f"GPU available: {'Yes' if gpu_available else 'No'}")
    logger.info(f"Cache directory: {cache_dir}")
    
    # Verification results
    logger.info(f"\nVERIFICATION RESULTS:")
    for component, status in verification_results.items():
        status_str = "✓ PASS" if status else "✗ FAIL"
        logger.info(f"  {component}: {status_str}")
    
    # Calculate cache size
    try:
        cache_size = sum(f.stat().st_size for f in cache_dir.rglob('*') if f.is_file())
        cache_size_gb = cache_size / (1024**3)
        logger.info(f"Total cache size: {cache_size_gb:.2f} GB")
    except Exception as e:
        logger.warning(f"Could not calculate cache size: {e}")
    
    all_verified = all(verification_results.values())
    
    if all_verified:
        logger.info(f"\n🎉 All models downloaded and verified successfully!")
        logger.info("You can now run the AnimateDiff backend.")
        return True
    else:
        logger.error(f"\n❌ Some models failed verification. Please check the logs above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)