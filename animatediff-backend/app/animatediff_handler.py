import torch
import os
import time
import logging
from typing import Optional, Dict, Any
from pathlib import Path
import numpy as np
from PIL import Image
import imageio
import asyncio
from .models import VideoGenerationRequest, Job
from .queue_manager import queue_manager

# Handle diffusers imports with error handling
try:
    from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
    from diffusers.utils import export_to_video
    DIFFUSERS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Diffusers import failed: {e}")
    DIFFUSERS_AVAILABLE = False

# Disable xFormers if it's causing issues
os.environ["XFORMERS_DISABLED"] = "1"

logger = logging.getLogger(__name__)

class AnimateDiffHandler:
    def __init__(self):
        self.device = self._get_device()
        self.pipeline = None
        self.model_loaded = False
        self.model_path = "SG161222/Realistic_Vision_V4.0_noVAE"
        self.motion_adapter_path = "guoyww/animatediff-motion-adapter-v1-5-2"
        self.output_dir = Path(os.getenv("OUTPUT_DIR", "./outputs"))
        self.output_dir.mkdir(exist_ok=True)
        
        logger.info(f"AnimateDiff handler initialized with device: {self.device}")
    
    def _get_device(self) -> str:
        """Determine the best device to use"""
        if os.getenv("FORCE_CPU", "false").lower() == "true":
            return "cpu"
        
        if torch.cuda.is_available():
            return "cuda"
        
        return "cpu"
    
    async def load_model(self) -> bool:
        """Load the AnimateDiff model"""
        if self.model_loaded:
            return True
        
        if not DIFFUSERS_AVAILABLE:
            logger.error("Diffusers not available - cannot load models")
            return False
        
        try:
            logger.info("Loading AnimateDiff model...")
            
            # Load in a separate thread to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._load_model_sync)
            
            self.model_loaded = True
            logger.info("AnimateDiff model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load AnimateDiff model: {e}")
            return False
    
    def _load_model_sync(self):
        """Synchronous model loading"""
        try:
            from diffusers import AnimateDiffPipeline, MotionAdapter
            
            # Load motion adapter
            adapter = MotionAdapter.from_pretrained(
                self.motion_adapter_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            )
            
            # Load pipeline
            self.pipeline = AnimateDiffPipeline.from_pretrained(
                self.model_path,
                motion_adapter=adapter,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            )
            
            # Set scheduler with proper configuration
            try:
                # Try with fixed configuration first
                self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                    self.pipeline.scheduler.config,
                    algorithm_type="sde-dpmsolver++",
                    solver_order=2,
                    final_sigmas_type="sigma_min"
                )
                logger.info("Using DPMSolverMultistepScheduler with sigma_min")
            except Exception as scheduler_error:
                logger.warning(f"DPMSolver failed: {scheduler_error}, trying alternative scheduler")
                # Fall back to a simpler scheduler that should work
                from diffusers import EulerDiscreteScheduler
                self.pipeline.scheduler = EulerDiscreteScheduler.from_config(
                    self.pipeline.scheduler.config
                )
                logger.info("Using EulerDiscreteScheduler as fallback")
            
            # Enable optimizations (skip xFormers-dependent ones)
            if self.device == "cuda":
                try:
                    self.pipeline.enable_vae_slicing()
                    self.pipeline.enable_model_cpu_offload()
                except Exception as e:
                    logger.warning(f"Could not enable optimizations: {e}")
            
            self.pipeline = self.pipeline.to(self.device)
            
        except Exception as e:
            logger.error(f"Error in model loading: {e}")
            raise
    
    async def generate_video(self, job: Job) -> Dict[str, Any]:
        """Generate video from job request"""
        if not self.model_loaded:
            await self.load_model()
        
        if not self.model_loaded:
            raise RuntimeError("Failed to load model")
        
        request = job.request
        job_id = job.job_id
        
        try:
            # Update progress
            await queue_manager.update_job_progress(job_id, 0.1, "Preparing generation...")
            
            # Set seed if provided
            if request.seed != -1:
                generator = torch.Generator(device=self.device).manual_seed(request.seed)
            else:
                generator = None
            
            # Prepare generation parameters
            generation_params = {
                "prompt": request.prompt,
                "negative_prompt": request.negative_prompt,
                "num_inference_steps": request.steps,
                "guidance_scale": request.guidance_scale,
                "width": request.width,
                "height": request.height,
                "num_frames": request.duration,
                "generator": generator,
            }
            
            # Update progress
            await queue_manager.update_job_progress(job_id, 0.2, "Starting generation...")
            
            # Generate video
            start_time = time.time()
            
            # Run generation in executor to avoid blocking
            loop = asyncio.get_event_loop()
            frames = await loop.run_in_executor(
                None, 
                lambda: self._generate_frames(generation_params, job_id)
            )
            
            # Apply any progress updates that were collected during generation
            if hasattr(self, '_progress_updates') and self._progress_updates:
                # Use the last progress update
                last_progress, last_message = self._progress_updates[-1]
                await queue_manager.update_job_progress(job_id, last_progress, last_message)
            
            generation_time = time.time() - start_time
            
            # Update progress
            await queue_manager.update_job_progress(job_id, 0.8, "Saving video...")
            
            # Save video
            video_path = self.output_dir / f"{job_id}.mp4"
            thumbnail_path = self.output_dir / f"{job_id}_thumb.jpg"
            
            # Export to video
            export_to_video(frames, str(video_path), fps=8)
            
            # Create thumbnail from first frame
            if frames:
                thumbnail = Image.fromarray(frames[0])
                thumbnail.save(thumbnail_path, "JPEG", quality=85)
            
            # Update progress
            await queue_manager.update_job_progress(job_id, 1.0, "Generation complete!")
            
            return {
                "video_path": str(video_path),
                "thumbnail_path": str(thumbnail_path),
                "generation_time": generation_time,
                "metadata": {
                    "frames": len(frames),
                    "fps": 8,
                    "duration": len(frames) / 8,
                    "resolution": f"{request.width}x{request.height}"
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating video for job {job_id}: {e}")
            raise
    
    def _generate_frames(self, params: Dict[str, Any], job_id: str) -> list:
        """Generate frames synchronously with progress updates"""
        try:
            # Store progress updates to apply later
            self._progress_updates = []
            
            # Create callback for progress updates (synchronous only)
            def callback(step: int, timestep: int, latents: torch.FloatTensor):
                progress = 0.2 + (step / params["num_inference_steps"]) * 0.6
                message = f"Generating frame {step}/{params['num_inference_steps']}"
                # Store the update instead of trying to run async code
                self._progress_updates.append((progress, message))
                logger.info(f"Generation progress: {progress:.1%} - {message}")
            
            # Generate with callback
            result = self.pipeline(
                callback=callback,
                callback_steps=1,
                **params
            )
            
            return result.frames[0]
            
        except Exception as e:
            logger.error(f"Error in frame generation: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            "model_loaded": self.model_loaded,
            "device": self.device,
            "model_path": self.model_path,
            "motion_adapter_path": self.motion_adapter_path,
            "gpu_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "gpu_memory": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None
        }
    
    def cleanup(self):
        """Clean up resources"""
        if self.pipeline:
            del self.pipeline
            self.pipeline = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.model_loaded = False
        logger.info("AnimateDiff handler cleaned up")

# Global handler instance
animatediff_handler = AnimateDiffHandler()