"""
MusicGen FastAPI Backend with CUDA GPU Acceleration

A FastAPI application that serves MusicGen AI music generation with a web interface.
Supports multiple model sizes, CUDA GPU acceleration, and generates audio files from text prompts.
Features intelligent GPU/CPU device management with automatic fallback capabilities.
"""

import os
import logging
import uuid
import gc
import time
import platform
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
import asyncio
from concurrent.futures import ThreadPoolExecutor
import queue
import threading

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import torch
from transformers import MusicgenForConditionalGeneration, AutoProcessor
import scipy.io.wavfile
import numpy as np

# Try to import GPU monitoring libraries
try:
    import GPUtil
    GPU_MONITORING_AVAILABLE = True
except ImportError:
    GPU_MONITORING_AVAILABLE = False
    
try:
    import psutil
    SYSTEM_MONITORING_AVAILABLE = True
except ImportError:
    SYSTEM_MONITORING_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="MusicGen API",
    description="AI Music Generation using MusicGen",
    version="1.0.0"
)

# Add CORS middleware for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model management
current_model = None
current_processor = None
current_model_name = None
current_device = None
device_info = {}
generation_queue = queue.Queue()
executor = ThreadPoolExecutor(max_workers=1)

# Performance tracking
generation_stats = {
    "total_generations": 0,
    "gpu_generations": 0,
    "cpu_generations": 0,
    "average_gpu_time": 0.0,
    "average_cpu_time": 0.0,
    "gpu_memory_peak": 0.0,
    "last_generation_time": 0.0,
    "cuda_errors": 0,
    "fallback_count": 0
}

# Available models
AVAILABLE_MODELS = {
    "small": "facebook/musicgen-small",
    "medium": "facebook/musicgen-medium",
    "large": "facebook/musicgen-large"
}

# Directories
GENERATED_MUSIC_DIR = Path("generated_music")
MODELS_DIR = Path("models")
STATIC_DIR = Path("static")

# Ensure directories exist
GENERATED_MUSIC_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)


class MusicGenerationRequest(BaseModel):
    """Request model for music generation"""
    prompt: str = Field(..., description="Text description of the music to generate")
    duration: float = Field(default=30.0, ge=2.0, le=300.0, description="Duration in seconds (2-300)")
    model: str = Field(default="small", description="Model size to use")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")
    force_cpu: bool = Field(default=False, description="Force CPU processing (disable GPU)")
    preferred_device: Optional[str] = Field(default=None, description="Preferred device ID for multi-GPU systems")


class MusicGenerationResponse(BaseModel):
    """Response model for music generation"""
    job_id: str
    status: str
    message: str
    file_path: Optional[str] = None
    duration: Optional[float] = None
    model_used: Optional[str] = None
    device_used: Optional[str] = None
    generation_time: Optional[float] = None
    gpu_memory_used: Optional[float] = None
    created_at: str


class GenerationJob:
    """Class to track generation jobs with performance metrics"""
    def __init__(self, job_id: str, request: MusicGenerationRequest):
        self.job_id = job_id
        self.request = request
        self.status = "queued"
        self.message = "Job queued for processing"
        self.file_path = None
        self.created_at = datetime.now().isoformat()
        self.error = None
        self.device_used = None
        self.generation_time = None
        self.gpu_memory_used = None
        self.start_time = None
        self.end_time = None


# Store active jobs
active_jobs = {}


def get_nvidia_driver_version() -> Optional[str]:
    """Get NVIDIA driver version using nvidia-smi"""
    try:
        if platform.system() == "Windows":
            result = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader,nounits"], 
                                  capture_output=True, text=True, timeout=10)
        else:
            result = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader,nounits"], 
                                  capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            return result.stdout.strip().split('\n')[0]
    except (subprocess.SubprocessError, FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def get_cuda_memory_info(device_id: int = 0) -> Dict[str, float]:
    """
    Get CUDA memory information for specified device.
    
    Args:
        device_id: CUDA device ID
        
    Returns:
        Dictionary with memory info in GB
    """
    try:
        if torch.cuda.is_available() and device_id < torch.cuda.device_count():
            torch.cuda.set_device(device_id)
            total_memory = torch.cuda.get_device_properties(device_id).total_memory / (1024**3)
            allocated_memory = torch.cuda.memory_allocated(device_id) / (1024**3)
            cached_memory = torch.cuda.memory_reserved(device_id) / (1024**3)
            free_memory = total_memory - allocated_memory
            
            return {
                "total": round(total_memory, 2),
                "allocated": round(allocated_memory, 2),
                "cached": round(cached_memory, 2),
                "free": round(free_memory, 2)
            }
    except Exception as e:
        logger.warning(f"Failed to get CUDA memory info: {e}")
    
    return {"total": 0, "allocated": 0, "cached": 0, "free": 0}


def check_cuda_compatibility() -> Dict[str, Any]:
    """
    Comprehensive CUDA compatibility and capability check.
    
    Returns:
        Dictionary with detailed CUDA information
    """
    cuda_info = {
        "cuda_available": False,
        "cuda_version": None,
        "cudnn_version": None,
        "device_count": 0,
        "devices": [],
        "driver_version": None,
        "pytorch_cuda_version": None,
        "compute_capabilities": [],
        "recommended_settings": {}
    }
    
    try:
        # Basic CUDA availability
        cuda_info["cuda_available"] = torch.cuda.is_available()
        cuda_info["pytorch_cuda_version"] = torch.version.cuda if hasattr(torch.version, 'cuda') else None
        
        if torch.cuda.is_available():
            # CUDA version and device count
            cuda_info["cuda_version"] = torch.version.cuda
            cuda_info["device_count"] = torch.cuda.device_count()
            cuda_info["driver_version"] = get_nvidia_driver_version()
            
            # cuDNN version
            if hasattr(torch.backends.cudnn, 'version'):
                cuda_info["cudnn_version"] = torch.backends.cudnn.version()
            
            # Device information
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                memory_info = get_cuda_memory_info(i)
                
                device_info = {
                    "id": i,
                    "name": props.name,
                    "compute_capability": f"{props.major}.{props.minor}",
                    "total_memory_gb": round(props.total_memory / (1024**3), 2),
                    "memory_info": memory_info,
                    "multiprocessor_count": props.multi_processor_count,
                    "is_integrated": getattr(props, 'integrated', False),  # Some PyTorch versions don't have this
                    "supports_fp16": props.major >= 6,  # Pascal and newer
                    "supports_bf16": props.major >= 8,  # Ampere and newer
                }
                
                cuda_info["devices"].append(device_info)
                cuda_info["compute_capabilities"].append(f"{props.major}.{props.minor}")
            
            # Recommendations based on available hardware
            if cuda_info["devices"]:
                best_device = max(cuda_info["devices"], key=lambda x: x["total_memory_gb"])
                
                cuda_info["recommended_settings"] = {
                    "use_fp16": best_device["supports_fp16"],
                    "use_bf16": best_device["supports_bf16"] and torch.__version__ >= "1.10",
                    "recommended_device": best_device["id"],
                    "batch_size_suggestion": min(4, max(1, int(best_device["total_memory_gb"] // 2))),
                    "memory_fraction": 0.8 if best_device["total_memory_gb"] > 6 else 0.9
                }
        
    except Exception as e:
        logger.error(f"Error checking CUDA compatibility: {e}")
        cuda_info["error"] = str(e)
    
    return cuda_info


def select_optimal_device(force_cpu: bool = False, preferred_device: Optional[str] = None) -> tuple[str, Dict[str, Any]]:
    """
    Select the optimal device for model inference.
    
    Args:
        force_cpu: Force CPU usage even if GPU is available
        preferred_device: Preferred GPU device ID (e.g., "cuda:0")
        
    Returns:
        Tuple of (device_string, device_info_dict)
    """
    global device_info
    
    if force_cpu:
        device_info = {
            "device": "cpu",
            "device_name": f"{platform.processor()} (Forced CPU)",
            "memory_gb": psutil.virtual_memory().total / (1024**3) if SYSTEM_MONITORING_AVAILABLE else "Unknown",
            "cuda_available": False,
            "reason": "CPU forced by user"
        }
        return "cpu", device_info
    
    cuda_info = check_cuda_compatibility()
    
    if not cuda_info["cuda_available"] or cuda_info["device_count"] == 0:
        device_info = {
            "device": "cpu",
            "device_name": platform.processor(),
            "memory_gb": psutil.virtual_memory().total / (1024**3) if SYSTEM_MONITORING_AVAILABLE else "Unknown",
            "cuda_available": False,
            "reason": "CUDA not available or no GPU devices found"
        }
        return "cpu", device_info
    
    # Select GPU device
    try:
        if preferred_device and preferred_device.startswith("cuda:"):
            device_id = int(preferred_device.split(":")[1])
            if device_id < cuda_info["device_count"]:
                selected_device = cuda_info["devices"][device_id]
            else:
                logger.warning(f"Preferred device {preferred_device} not available, using recommended device")
                selected_device = cuda_info["devices"][cuda_info["recommended_settings"]["recommended_device"]]
                device_id = cuda_info["recommended_settings"]["recommended_device"]
        else:
            device_id = cuda_info["recommended_settings"]["recommended_device"]
            selected_device = cuda_info["devices"][device_id]
        
        device_string = f"cuda:{device_id}"
        
        device_info = {
            "device": device_string,
            "device_id": device_id,
            "device_name": selected_device["name"],
            "compute_capability": selected_device["compute_capability"],
            "total_memory_gb": selected_device["total_memory_gb"],
            "memory_info": selected_device["memory_info"],
            "cuda_available": True,
            "supports_fp16": selected_device["supports_fp16"],
            "supports_bf16": selected_device["supports_bf16"],
            "cuda_version": cuda_info["cuda_version"],
            "driver_version": cuda_info["driver_version"],
            "reason": "Optimal GPU device selected"
        }
        
        logger.info(f"Selected GPU device: {device_string} ({selected_device['name']})")
        return device_string, device_info
        
    except Exception as e:
        logger.error(f"Error selecting GPU device: {e}")
        # Fallback to CPU
        device_info = {
            "device": "cpu", 
            "device_name": platform.processor(),
            "memory_gb": psutil.virtual_memory().total / (1024**3) if SYSTEM_MONITORING_AVAILABLE else "Unknown",
            "cuda_available": False,
            "reason": f"GPU selection failed: {str(e)}"
        }
        return "cpu", device_info


def cleanup_gpu_memory():
    """Clean up GPU memory and cache"""
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()
        except Exception as e:
            logger.warning(f"GPU memory cleanup failed: {e}")


def load_model_with_fallback(model_name: str, force_cpu: bool = False, 
                           preferred_device: Optional[str] = None) -> tuple[MusicgenForConditionalGeneration, AutoProcessor, str]:
    """
    Load MusicGen model with intelligent device selection and fallback.
    
    Args:
        model_name: Name of the model to load
        force_cpu: Force CPU usage
        preferred_device: Preferred device for loading
        
    Returns:
        Tuple of (model, processor, device_used)
        
    Raises:
        RuntimeError: If model loading fails on all devices
    """
    global current_device, generation_stats
    
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Model {model_name} not available. Choose from: {list(AVAILABLE_MODELS.keys())}")
    
    model_path = AVAILABLE_MODELS[model_name]
    logger.info(f"Loading model: {model_path}")
    
    # Select optimal device
    device_str, device_info_dict = select_optimal_device(force_cpu, preferred_device)
    current_device = device_str
    
    # Clean up existing memory
    cleanup_gpu_memory()
    
    processor = None
    model = None
    
    try:
        # Load processor (always CPU)
        logger.info("Loading processor...")
        processor = AutoProcessor.from_pretrained(model_path)
        
        # Determine optimal model loading strategy
        is_gpu = device_str.startswith("cuda")
        
        if is_gpu:
            try:
                logger.info(f"Loading model on GPU: {device_str}")
                
                # Get memory info before loading
                device_id = int(device_str.split(":")[1]) if ":" in device_str else 0
                memory_before = get_cuda_memory_info(device_id)
                
                # Determine optimal precision
                use_fp16 = device_info_dict.get("supports_fp16", False)
                use_bf16 = device_info_dict.get("supports_bf16", False) and torch.__version__ >= "1.10"
                
                if use_bf16:
                    torch_dtype = torch.bfloat16
                    logger.info("Using bfloat16 precision for optimal performance")
                elif use_fp16:
                    torch_dtype = torch.float16
                    logger.info("Using float16 precision for memory efficiency")
                else:
                    torch_dtype = torch.float32
                    logger.info("Using float32 precision")
                
                # Load model with optimal settings and force safetensors
                try:
                    model = MusicgenForConditionalGeneration.from_pretrained(
                        model_path,
                        torch_dtype=torch_dtype,
                        device_map="auto",
                        low_cpu_mem_usage=True,
                        use_safetensors=True,  # Force safetensors to avoid torch.load security issue
                        trust_remote_code=False  # Security setting
                    )
                except Exception as safetensor_error:
                    logger.warning(f"Safetensors loading failed, trying with legacy loader: {safetensor_error}")
                    # Fallback to legacy loading with weights_only
                    import os
                    os.environ["TRANSFORMERS_TRUST_REMOTE_CODE"] = "False"
                    model = MusicgenForConditionalGeneration.from_pretrained(
                        model_path,
                        torch_dtype=torch_dtype,
                        device_map="auto",
                        low_cpu_mem_usage=True,
                        use_safetensors=False
                    )
                
                # Verify model is on GPU
                model_device = next(model.parameters()).device
                if not str(model_device).startswith("cuda"):
                    raise RuntimeError(f"Model loaded on {model_device} instead of requested {device_str}")
                
                # Get memory info after loading
                memory_after = get_cuda_memory_info(device_id)
                memory_used = memory_after["allocated"] - memory_before["allocated"]
                
                logger.info(f"Model loaded successfully on {device_str}")
                logger.info(f"GPU memory usage: {memory_used:.2f}GB allocated, {memory_after['free']:.2f}GB free")
                
                # Enable optimizations
                if hasattr(torch.backends.cudnn, 'benchmark'):
                    torch.backends.cudnn.benchmark = True
                if hasattr(torch.backends.cudnn, 'enabled'):
                    torch.backends.cudnn.enabled = True
                
                return model, processor, device_str
                
            except torch.cuda.OutOfMemoryError as e:
                logger.error(f"CUDA out of memory: {e}")
                generation_stats["cuda_errors"] += 1
                generation_stats["fallback_count"] += 1
                cleanup_gpu_memory()
                
                # Fallback to CPU
                logger.warning("Falling back to CPU due to GPU memory error")
                device_str = "cpu"
                current_device = "cpu"
                
            except Exception as e:
                logger.error(f"GPU model loading failed: {e}")
                generation_stats["cuda_errors"] += 1
                generation_stats["fallback_count"] += 1
                cleanup_gpu_memory()
                
                # Fallback to CPU
                logger.warning("Falling back to CPU due to GPU error")
                device_str = "cpu"
                current_device = "cpu"
        
        # Load on CPU (either forced or fallback)
        if device_str == "cpu":
            logger.info("Loading model on CPU")
            try:
                model = MusicgenForConditionalGeneration.from_pretrained(
                    model_path,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                    use_safetensors=True,
                    trust_remote_code=False
                )
            except Exception as safetensor_error:
                logger.warning(f"Safetensors loading failed for CPU, using legacy: {safetensor_error}")
                import os
                os.environ["TRANSFORMERS_TRUST_REMOTE_CODE"] = "False"
                model = MusicgenForConditionalGeneration.from_pretrained(
                    model_path,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                    use_safetensors=False
                )
            model = model.to("cpu")
            logger.info("Model loaded successfully on CPU")
        
        return model, processor, device_str
        
    except Exception as e:
        # Clean up partial loads
        if model is not None:
            del model
        if processor is not None:
            del processor
        cleanup_gpu_memory()
        
        logger.error(f"Failed to load model {model_name}: {str(e)}")
        raise RuntimeError(f"Model loading failed on all devices: {str(e)}")


# Backwards compatibility wrapper
def load_model(model_name: str) -> tuple[MusicgenForConditionalGeneration, AutoProcessor]:
    """
    Legacy wrapper for load_model_with_fallback.
    
    Args:
        model_name: Name of the model to load
        
    Returns:
        Tuple of (model, processor)
        
    Raises:
        RuntimeError: If model loading fails
    """
    model, processor, _ = load_model_with_fallback(model_name)
    return model, processor


def cleanup_cuda_memory() -> None:
    """Clean up CUDA memory to prevent fragmentation during long generations."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()


def generate_single_audio_chunk(
    model, inputs: Dict[str, torch.Tensor], generation_kwargs: Dict[str, Any], 
    is_gpu: bool, device_id: Optional[int] = None
) -> torch.Tensor:
    """
    Generate a single audio chunk with proper memory management.
    
    Args:
        model: The MusicGen model
        inputs: Processed inputs for the model
        generation_kwargs: Generation parameters
        is_gpu: Whether using GPU
        device_id: CUDA device ID if using GPU
        
    Returns:
        Generated audio tensor
    """
    try:
        with torch.no_grad():
            # GPU-specific optimizations for single chunk
            if is_gpu and hasattr(model.config, 'use_memory_efficient_attention'):
                model.config.use_memory_efficient_attention = True
            
            generation_start = time.time()
            audio_values = model.generate(**inputs, **generation_kwargs)
            generation_time = time.time() - generation_start
            
            logger.info(f"Single chunk generated in {generation_time:.2f}s, shape: {audio_values.shape}")
            
            # Clean up memory after generation
            if is_gpu:
                cleanup_cuda_memory()
            
            return audio_values
            
    except Exception as e:
        logger.error(f"Single chunk generation failed: {e}")
        raise


def generate_long_audio_chunked(
    model, processor, inputs: Dict[str, torch.Tensor], job: 'GenerationJob', 
    is_gpu: bool, device_id: Optional[int] = None
) -> torch.Tensor:
    """
    Generate long audio using chunked approach with seamless concatenation.
    
    This function implements sliding window generation to handle long durations
    that would otherwise cause memory issues or quality degradation.
    
    Args:
        model: The MusicGen model
        processor: The audio processor
        inputs: Processed inputs for the model
        job: Generation job containing request details
        is_gpu: Whether using GPU
        device_id: CUDA device ID if using GPU
        
    Returns:
        Concatenated audio tensor for the full duration
    """
    CHUNK_DURATION = 30.0  # Duration of each chunk in seconds
    OVERLAP_DURATION = 5.0  # Overlap between chunks for seamless blending
    
    total_duration = job.request.duration
    num_chunks = int(np.ceil((total_duration - OVERLAP_DURATION) / (CHUNK_DURATION - OVERLAP_DURATION)))
    
    logger.info(f"Generating {total_duration}s audio in {num_chunks} chunks of {CHUNK_DURATION}s each")
    
    audio_chunks = []
    conditioning_audio = None  # Store previous chunk for conditioning
    
    for chunk_idx in range(num_chunks):
        try:
            # Calculate chunk timing
            chunk_start_time = chunk_idx * (CHUNK_DURATION - OVERLAP_DURATION)
            remaining_duration = total_duration - chunk_start_time
            actual_chunk_duration = min(CHUNK_DURATION, remaining_duration)
            
            # Update job progress
            progress_percent = (chunk_idx / num_chunks) * 100
            job.message = f"Generating chunk {chunk_idx + 1}/{num_chunks} ({progress_percent:.1f}%)"
            logger.info(f"Generating chunk {chunk_idx + 1}/{num_chunks}: {actual_chunk_duration:.1f}s")
            
            # Calculate tokens for this chunk
            chunk_max_tokens = int(actual_chunk_duration * 50)
            if chunk_max_tokens % 4 != 0:
                chunk_max_tokens = ((chunk_max_tokens // 4) + 1) * 4
            
            # Prepare generation parameters for this chunk
            chunk_kwargs = {
                "max_new_tokens": chunk_max_tokens,
                "do_sample": True,
                "temperature": 1.0,  # Consistent temperature for quality
            }
            
            # For continuation chunks, use conditioning from previous audio
            chunk_inputs = inputs.copy()
            if conditioning_audio is not None and chunk_idx > 0:
                # Use the last part of previous chunk as conditioning
                # This helps maintain musical coherence across chunks
                conditioning_length = min(conditioning_audio.shape[-1], 16000)  # ~0.5s at 32kHz
                conditioning_tensor = conditioning_audio[..., -conditioning_length:].contiguous()
                
                # Note: Actual conditioning implementation depends on MusicGen architecture
                # For now, we rely on the text prompt consistency
                logger.debug(f"Using conditioning from previous chunk: {conditioning_tensor.shape}")
            
            # Memory management before generation
            if is_gpu:
                memory_before = get_cuda_memory_info(device_id)
                logger.debug(f"GPU memory before chunk {chunk_idx + 1}: {memory_before['allocated']:.2f}GB")
                
                # Clean memory if getting close to limit
                if memory_before["allocated"] / memory_before["total"] > 0.8:
                    logger.warning(f"GPU memory usage high ({memory_before['allocated']:.1f}GB), cleaning cache")
                    cleanup_cuda_memory()
            
            # Generate this chunk
            with torch.no_grad():
                if is_gpu and hasattr(model.config, 'use_memory_efficient_attention'):
                    model.config.use_memory_efficient_attention = True
                
                chunk_audio = model.generate(**chunk_inputs, **chunk_kwargs)
                
                # Convert to float32 immediately to save memory
                chunk_audio = chunk_audio.float()
                
                # Log chunk details
                logger.info(f"Chunk {chunk_idx + 1} generated: shape {chunk_audio.shape}, "
                          f"duration ~{chunk_audio.shape[-1] / 32000:.1f}s")
                
                # Store for next iteration conditioning
                conditioning_audio = chunk_audio.clone()
                
                # Memory cleanup after generation
                if is_gpu:
                    cleanup_cuda_memory()
                    memory_after = get_cuda_memory_info(device_id)
                    logger.debug(f"GPU memory after chunk {chunk_idx + 1}: {memory_after['allocated']:.2f}GB")
            
            # Handle overlap and concatenation
            if chunk_idx == 0:
                # First chunk - use as-is
                audio_chunks.append(chunk_audio)
            else:
                # Subsequent chunks - handle overlap for seamless blending
                overlap_samples = int(OVERLAP_DURATION * 32000)  # Assume 32kHz
                
                if len(audio_chunks) > 0 and audio_chunks[-1].shape[-1] > overlap_samples:
                    # Remove overlap from end of previous chunk
                    audio_chunks[-1] = audio_chunks[-1][..., :-overlap_samples]
                
                # Remove overlap from beginning of current chunk if it's long enough
                if chunk_audio.shape[-1] > overlap_samples:
                    chunk_audio = chunk_audio[..., overlap_samples:]
                
                audio_chunks.append(chunk_audio)
            
            # Clean up chunk variables to save memory
            del chunk_audio
            if is_gpu:
                cleanup_cuda_memory()
        
        except Exception as e:
            logger.error(f"Failed to generate chunk {chunk_idx + 1}: {e}")
            if is_gpu and "memory" in str(e).lower():
                logger.warning("GPU memory error in chunked generation, attempting recovery")
                cleanup_cuda_memory()
                
                # Try with reduced chunk size
                reduced_duration = CHUNK_DURATION * 0.7
                reduced_tokens = int(reduced_duration * 50)
                if reduced_tokens % 4 != 0:
                    reduced_tokens = ((reduced_tokens // 4) + 1) * 4
                
                reduced_kwargs = {
                    "max_new_tokens": reduced_tokens,
                    "do_sample": True,
                    "temperature": 1.0,
                }
                
                logger.info(f"Retrying chunk {chunk_idx + 1} with reduced size: {reduced_duration:.1f}s")
                
                try:
                    with torch.no_grad():
                        chunk_audio = model.generate(**chunk_inputs, **reduced_kwargs)
                        chunk_audio = chunk_audio.float()
                        audio_chunks.append(chunk_audio)
                        logger.info(f"Chunk {chunk_idx + 1} recovered with reduced size")
                except Exception as retry_error:
                    logger.error(f"Chunk recovery failed: {retry_error}")
                    raise
            else:
                raise
    
    # Concatenate all chunks
    try:
        if len(audio_chunks) == 0:
            raise ValueError("No audio chunks were generated successfully")
        
        logger.info(f"Concatenating {len(audio_chunks)} audio chunks")
        
        # Ensure all chunks have the same shape except for the last dimension
        reference_shape = audio_chunks[0].shape[:-1]
        for i, chunk in enumerate(audio_chunks):
            if chunk.shape[:-1] != reference_shape:
                logger.warning(f"Chunk {i} shape mismatch: {chunk.shape} vs expected {reference_shape}")
                # Reshape to match
                if len(chunk.shape) == 3 and len(reference_shape) == 3:
                    audio_chunks[i] = chunk.view(reference_shape + (-1,))
        
        # Concatenate along the time dimension
        final_audio = torch.cat(audio_chunks, dim=-1)
        
        # Log final statistics
        total_samples = final_audio.shape[-1]
        actual_duration = total_samples / 32000  # Assume 32kHz
        logger.info(f"Final concatenated audio: shape {final_audio.shape}, "
                   f"duration {actual_duration:.2f}s (requested {total_duration:.2f}s)")
        
        # Clean up chunk list to free memory
        del audio_chunks
        if is_gpu:
            cleanup_cuda_memory()
        
        return final_audio
        
    except Exception as e:
        logger.error(f"Failed to concatenate audio chunks: {e}")
        raise


def generate_music_sync(job: GenerationJob) -> None:
    """
    Enhanced synchronous music generation with CUDA support and performance monitoring.
    
    Args:
        job: Generation job to process
    """
    global current_model, current_processor, current_model_name, current_device, generation_stats
    
    job.start_time = time.time()
    generation_start_time = time.time()
    gpu_memory_peak = 0.0
    
    try:
        job.status = "processing"
        job.message = "Initializing generation..."
        logger.info(f"Starting generation for job {job.job_id}")
        
        # Determine device preferences from request
        force_cpu = getattr(job.request, 'force_cpu', False)
        preferred_device = getattr(job.request, 'preferred_device', None)
        
        # Load model if needed or if device preferences changed
        model_needs_reload = (
            current_model is None or 
            current_model_name != job.request.model or
            (force_cpu and current_device != "cpu") or
            (preferred_device and current_device != preferred_device)
        )
        
        if model_needs_reload:
            job.message = f"Loading {job.request.model} model on optimal device..."
            
            # Clean up existing model
            if current_model is not None:
                del current_model
                current_model = None
                cleanup_gpu_memory()
            
            # Load model with new settings
            current_model, current_processor, device_used = load_model_with_fallback(
                job.request.model, force_cpu, preferred_device
            )
            current_model_name = job.request.model
            current_device = device_used
            job.device_used = device_used
            
            logger.info(f"Model loaded on device: {device_used}")
        else:
            job.device_used = current_device
            logger.info(f"Using existing model on device: {current_device}")
        
        # Monitor GPU memory before generation
        is_gpu = current_device.startswith("cuda")
        if is_gpu:
            device_id = int(current_device.split(":")[1]) if ":" in current_device else 0
            memory_before = get_cuda_memory_info(device_id)
        
        # Prepare generation parameters
        job.message = "Preparing inputs..."
        inputs = current_processor(
            text=[job.request.prompt],
            padding=True,
            return_tensors="pt"
        )
        
        # Move inputs to same device as model
        device = next(current_model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Set random seed if provided
        if job.request.seed is not None:
            torch.manual_seed(job.request.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(job.request.seed)
                torch.cuda.manual_seed_all(job.request.seed)
        
        job.message = "Generating music with AI..."
        
        # Start generation timing
        generation_actual_start = time.time()
        
        # Determine if we need chunked generation for long durations
        CHUNK_DURATION = 30.0  # Maximum chunk size in seconds
        use_chunking = job.request.duration > CHUNK_DURATION
        
        try:
            if use_chunking:
                logger.info(f"Using chunked generation for {job.request.duration}s duration")
                audio_values = generate_long_audio_chunked(
                    current_model, current_processor, inputs, job, is_gpu, device_id if is_gpu else None
                )
            else:
                # Standard single-shot generation for short durations
                max_new_tokens = int(job.request.duration * 50)
                if max_new_tokens % 4 != 0:
                    max_new_tokens = ((max_new_tokens // 4) + 1) * 4
                
                generation_kwargs = {
                    "max_new_tokens": max_new_tokens,
                    "do_sample": True,
                }
                
                audio_values = generate_single_audio_chunk(
                    current_model, inputs, generation_kwargs, is_gpu, device_id if is_gpu else None
                )
        
        except Exception as e:
            logger.error(f"Generation error: {e}")
            if is_gpu and not force_cpu:
                logger.warning("Retrying on CPU due to GPU error")
                generation_stats["fallback_count"] += 1
                
                # Simple CPU fallback
                del current_model
                current_model, current_processor, device_used = load_model_with_fallback(
                    job.request.model, force_cpu=True
                )
                current_device = device_used
                job.device_used = device_used
                
                # Retry on CPU
                inputs = {k: v.to("cpu") for k, v in inputs.items()}
                
                if use_chunking:
                    audio_values = generate_long_audio_chunked(
                        current_model, current_processor, inputs, job, False, None
                    )
                else:
                    audio_values = generate_single_audio_chunk(
                        current_model, inputs, generation_kwargs, False, None
                    )
            else:
                raise
        
        # Monitor peak GPU memory usage
        if is_gpu:
            memory_during = get_cuda_memory_info(device_id)
            gpu_memory_peak = memory_during["allocated"]
            job.gpu_memory_used = gpu_memory_peak
        
        generation_time = time.time() - generation_actual_start
        job.generation_time = generation_time
        
        job.message = "Processing generated audio..."
        
        # Convert to numpy - handle different tensor types and shapes
        logger.info(f"Generated audio tensor shape: {audio_values.shape}, dtype: {audio_values.dtype}")
        
        # Convert to float32 first (numpy compatible), then extract audio
        audio_values = audio_values.float()  # Convert any precision to float32
        
        # Simple extraction based on shape
        if len(audio_values.shape) == 3:  # [batch, channels, time] - standard MusicGen
            audio_data = audio_values[0, 0].cpu().numpy()
        elif len(audio_values.shape) == 2:  # [batch, time]
            audio_data = audio_values[0].cpu().numpy()  
        else:  # [time] or other
            audio_data = audio_values.squeeze().cpu().numpy()
        
        # Normalize audio
        if np.max(np.abs(audio_data)) > 0:
            audio_data = audio_data / np.max(np.abs(audio_data))
        else:
            logger.warning("Generated audio is silent")
        
        # Generate filename with device info
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        device_suffix = "gpu" if is_gpu else "cpu"
        filename = f"music_{timestamp}_{job.job_id[:8]}_{device_suffix}.wav"
        file_path = GENERATED_MUSIC_DIR / filename
        
        # Get sample rate (MusicGen default is 32kHz)
        sample_rate = 32000
        if hasattr(current_model.config, 'audio_encoder') and hasattr(current_model.config.audio_encoder, 'sampling_rate'):
            sample_rate = current_model.config.audio_encoder.sampling_rate
        
        # Save audio file
        scipy.io.wavfile.write(file_path, sample_rate, (audio_data * 32767).astype(np.int16))
        
        # Update job status
        job.end_time = time.time()
        total_time = job.end_time - job.start_time
        job.status = "completed"
        job.message = "Music generation completed successfully"
        job.file_path = str(file_path)
        
        # Update generation statistics
        generation_stats["total_generations"] += 1
        if is_gpu:
            generation_stats["gpu_generations"] += 1
            generation_stats["average_gpu_time"] = (
                (generation_stats["average_gpu_time"] * (generation_stats["gpu_generations"] - 1) + generation_time) /
                generation_stats["gpu_generations"]
            )
            generation_stats["gpu_memory_peak"] = max(generation_stats["gpu_memory_peak"], gpu_memory_peak)
        else:
            generation_stats["cpu_generations"] += 1
            generation_stats["average_cpu_time"] = (
                (generation_stats["average_cpu_time"] * (generation_stats["cpu_generations"] - 1) + generation_time) /
                generation_stats["cpu_generations"]
            )
        
        generation_stats["last_generation_time"] = generation_time
        
        logger.info(f"Generation completed for job {job.job_id}: {file_path}")
        logger.info(f"Generation time: {generation_time:.2f}s, Total time: {total_time:.2f}s, Device: {job.device_used}")
        
        # Clean up GPU memory after generation
        if is_gpu:
            cleanup_gpu_memory()
        
    except Exception as e:
        job.end_time = time.time()
        job.status = "failed"
        job.message = f"Generation failed: {str(e)}"
        job.error = str(e)
        logger.error(f"Generation failed for job {job.job_id}: {str(e)}")
        
        # Clean up on error
        cleanup_gpu_memory()


@app.on_event("startup")
async def startup_event():
    """Initialize the application with enhanced CUDA support"""
    logger.info("Starting MusicGen API server with CUDA acceleration...")
    
    # Comprehensive system check
    cuda_info = check_cuda_compatibility()
    logger.info("=== System Information ===")
    logger.info(f"Platform: {platform.system()} {platform.release()}")
    logger.info(f"Python: {platform.python_version()}")
    logger.info(f"PyTorch: {torch.__version__}")
    
    if cuda_info["cuda_available"]:
        logger.info(f"CUDA: {cuda_info['cuda_version']}")
        logger.info(f"cuDNN: {cuda_info.get('cudnn_version', 'Not available')}")
        logger.info(f"GPU Driver: {cuda_info.get('driver_version', 'Unknown')}")
        logger.info(f"GPU Devices: {cuda_info['device_count']}")
        
        for i, device in enumerate(cuda_info["devices"]):
            logger.info(f"  GPU {i}: {device['name']} ({device['total_memory_gb']}GB, Compute {device['compute_capability']})")
    else:
        logger.info("CUDA: Not available - using CPU")
    
    # Set global device info
    global device_info
    device_str, device_info = select_optimal_device()
    logger.info(f"Selected device: {device_str} - {device_info.get('reason', 'No reason provided')}")
    
    # Preload small model for faster first generation
    try:
        logger.info("Preloading small model...")
        global current_model, current_processor, current_model_name, current_device
        current_model, current_processor, device_used = load_model_with_fallback("small")
        current_model_name = "small"
        current_device = device_used
        logger.info(f"Small model preloaded successfully on {device_used}")
        
        # Log memory usage if GPU
        if device_used.startswith("cuda"):
            device_id = int(device_used.split(":")[1]) if ":" in device_used else 0
            memory_info = get_cuda_memory_info(device_id)
            logger.info(f"GPU memory after model load: {memory_info['allocated']:.2f}GB allocated, {memory_info['free']:.2f}GB free")
            
    except Exception as e:
        logger.warning(f"Failed to preload model: {e}")
        logger.info("Model will be loaded on first generation request")
    
    logger.info("=== MusicGen API Ready ===")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown"""
    logger.info("Shutting down MusicGen API server...")
    
    global current_model, current_processor
    if current_model is not None:
        del current_model
        current_model = None
    if current_processor is not None:
        del current_processor
        current_processor = None
    
    cleanup_gpu_memory()
    logger.info("Cleanup completed")


@app.post("/generate-music", response_model=MusicGenerationResponse)
async def generate_music(request: MusicGenerationRequest, background_tasks: BackgroundTasks):
    """
    Generate music from text prompt.
    
    Args:
        request: Music generation parameters
        background_tasks: FastAPI background tasks
        
    Returns:
        Generation job information
    """
    try:
        # Validate model
        if request.model not in AVAILABLE_MODELS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model. Choose from: {list(AVAILABLE_MODELS.keys())}"
            )
        
        # Create job
        job_id = str(uuid.uuid4())
        job = GenerationJob(job_id, request)
        active_jobs[job_id] = job
        
        # Submit to thread pool
        future = executor.submit(generate_music_sync, job)
        
        logger.info(f"Submitted generation job {job_id} for prompt: '{request.prompt[:50]}...'")
        
        return MusicGenerationResponse(
            job_id=job_id,
            status=job.status,
            message=job.message,
            created_at=job.created_at
        )
        
    except Exception as e:
        logger.error(f"Failed to create generation job: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start generation: {str(e)}")


@app.get("/job/{job_id}", response_model=MusicGenerationResponse)
async def get_job_status(job_id: str):
    """
    Get the status of a generation job.
    
    Args:
        job_id: ID of the job to check
        
    Returns:
        Current job status and information
    """
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = active_jobs[job_id]
    
    return MusicGenerationResponse(
        job_id=job_id,
        status=job.status,
        message=job.message,
        file_path=job.file_path,
        duration=job.request.duration if job.status == "completed" else None,
        model_used=job.request.model if job.status == "completed" else None,
        device_used=job.device_used if job.status == "completed" else None,
        generation_time=job.generation_time if job.status == "completed" else None,
        gpu_memory_used=job.gpu_memory_used if job.status == "completed" else None,
        created_at=job.created_at
    )


@app.get("/download/{job_id}")
async def download_music(job_id: str):
    """
    Download generated music file.
    
    Args:
        job_id: ID of the completed job
        
    Returns:
        Audio file response
    """
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = active_jobs[job_id]
    
    if job.status != "completed" or not job.file_path:
        raise HTTPException(status_code=400, detail="Job not completed or file not available")
    
    file_path = Path(job.file_path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Generated file not found")
    
    return FileResponse(
        path=file_path,
        media_type="audio/wav",
        filename=file_path.name
    )


@app.get("/list-generated")
async def list_generated_music():
    """
    List all generated music files.
    
    Returns:
        List of generated music files with metadata
    """
    try:
        files = []
        for file_path in GENERATED_MUSIC_DIR.glob("*.wav"):
            stat = file_path.stat()
            files.append({
                "filename": file_path.name,
                "size": stat.st_size,
                "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "path": f"/download-file/{file_path.name}"
            })
        
        # Sort by creation time, newest first
        files.sort(key=lambda x: x["created"], reverse=True)
        return {"files": files}
        
    except Exception as e:
        logger.error(f"Failed to list files: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to list files")


@app.get("/download-file/{filename}")
async def download_file(filename: str):
    """
    Download a specific generated file by filename.
    
    Args:
        filename: Name of the file to download
        
    Returns:
        Audio file response
    """
    file_path = GENERATED_MUSIC_DIR / filename
    
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=file_path,
        media_type="audio/wav",
        filename=filename
    )


@app.get("/models")
async def get_available_models():
    """
    Get list of available models.
    
    Returns:
        List of available models with descriptions
    """
    return {
        "models": [
            {
                "name": "small",
                "description": "Fastest generation, lower quality (300M parameters)",
                "recommended_for": "Quick testing and iteration"
            },
            {
                "name": "medium",
                "description": "Balanced speed and quality (1.5B parameters)",
                "recommended_for": "General use"
            },
            {
                "name": "large",
                "description": "Highest quality, slower generation (3.3B parameters)",
                "recommended_for": "Final production"
            }
        ],
        "current_model": current_model_name
    }


@app.get("/system-info")
async def get_system_info():
    """
    Get comprehensive system and GPU information.
    
    Returns:
        Detailed system, CUDA, and performance information
    """
    global device_info, current_device, generation_stats
    
    # Get CUDA compatibility info
    cuda_info = check_cuda_compatibility()
    
    # Get current GPU memory if available
    current_gpu_memory = {}
    if current_device and current_device.startswith("cuda"):
        device_id = int(current_device.split(":")[1]) if ":" in current_device else 0
        current_gpu_memory = get_cuda_memory_info(device_id)
    
    # Get system memory info
    system_memory = {}
    if SYSTEM_MONITORING_AVAILABLE:
        memory = psutil.virtual_memory()
        system_memory = {
            "total_gb": round(memory.total / (1024**3), 2),
            "available_gb": round(memory.available / (1024**3), 2),
            "used_gb": round(memory.used / (1024**3), 2),
            "percent_used": memory.percent
        }
    
    return {
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "processor": platform.processor(),
            "python_version": platform.python_version()
        },
        "pytorch": {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if hasattr(torch.version, 'cuda') else None
        },
        "cuda": cuda_info,
        "current_device": {
            "device": current_device,
            "device_info": device_info,
            "memory": current_gpu_memory
        },
        "system_memory": system_memory,
        "model_info": {
            "current_model": current_model_name,
            "model_loaded": current_model is not None
        },
        "performance_stats": generation_stats,
        "application": {
            "generated_files_count": len(list(GENERATED_MUSIC_DIR.glob("*.wav"))),
            "monitoring_available": {
                "gpu": GPU_MONITORING_AVAILABLE,
                "system": SYSTEM_MONITORING_AVAILABLE
            }
        }
    }


@app.get("/gpu-status")
async def get_gpu_status():
    """
    Get real-time GPU status and memory information.
    
    Returns:
        Current GPU status and memory usage
    """
    if not torch.cuda.is_available():
        return {
            "cuda_available": False,
            "message": "CUDA not available"
        }
    
    gpu_status = []
    
    for device_id in range(torch.cuda.device_count()):
        memory_info = get_cuda_memory_info(device_id)
        props = torch.cuda.get_device_properties(device_id)
        
        # Try to get temperature and utilization if GPUtil is available
        additional_info = {}
        if GPU_MONITORING_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                if device_id < len(gpus):
                    gpu = gpus[device_id]
                    additional_info = {
                        "temperature": gpu.temperature,
                        "utilization_percent": gpu.load * 100,
                        "power_usage": getattr(gpu, 'powerDraw', None),
                        "power_limit": getattr(gpu, 'powerLimit', None)
                    }
            except Exception as e:
                logger.warning(f"Failed to get additional GPU info: {e}")
        
        device_status = {
            "device_id": device_id,
            "name": props.name,
            "memory": memory_info,
            "compute_capability": f"{props.major}.{props.minor}",
            "multiprocessor_count": props.multi_processor_count,
            "is_current_device": current_device == f"cuda:{device_id}",
            **additional_info
        }
        
        gpu_status.append(device_status)
    
    return {
        "cuda_available": True,
        "device_count": torch.cuda.device_count(),
        "devices": gpu_status,
        "current_device": current_device
    }


@app.get("/performance-metrics")
async def get_performance_metrics():
    """
    Get detailed performance metrics and generation statistics.
    
    Returns:
        Performance metrics and recommendations
    """
    global generation_stats
    
    # Calculate performance insights
    insights = []
    
    if generation_stats["total_generations"] > 0:
        gpu_ratio = generation_stats["gpu_generations"] / generation_stats["total_generations"]
        insights.append({
            "type": "info",
            "message": f"GPU usage: {gpu_ratio:.1%} of generations"
        })
        
        if generation_stats["cuda_errors"] > 0:
            error_ratio = generation_stats["cuda_errors"] / generation_stats["total_generations"]
            insights.append({
                "type": "warning",
                "message": f"CUDA errors in {error_ratio:.1%} of generations"
            })
        
        if generation_stats["fallback_count"] > 0:
            fallback_ratio = generation_stats["fallback_count"] / generation_stats["total_generations"]
            insights.append({
                "type": "warning",
                "message": f"CPU fallback used in {fallback_ratio:.1%} of generations"
            })
        
        # Performance comparison
        if generation_stats["gpu_generations"] > 0 and generation_stats["cpu_generations"] > 0:
            speedup = generation_stats["average_cpu_time"] / generation_stats["average_gpu_time"]
            insights.append({
                "type": "success",
                "message": f"GPU is {speedup:.1f}x faster than CPU on average"
            })
    
    # Recommendations
    recommendations = []
    
    if torch.cuda.is_available():
        if current_device == "cpu":
            recommendations.append({
                "type": "suggestion",
                "message": "Consider enabling GPU acceleration for faster generation"
            })
        
        if current_device and current_device.startswith("cuda"):
            device_id = int(current_device.split(":")[1]) if ":" in current_device else 0
            memory_info = get_cuda_memory_info(device_id)
            
            if memory_info["free"] < 2.0:
                recommendations.append({
                    "type": "warning",
                    "message": "Low GPU memory available. Consider using smaller models or clearing cache."
                })
            
            if generation_stats["gpu_memory_peak"] > memory_info["total"] * 0.9:
                recommendations.append({
                    "type": "warning",
                    "message": "GPU memory usage is very high. Consider reducing generation duration or using smaller models."
                })
    else:
        recommendations.append({
            "type": "info",
            "message": "Install CUDA-enabled PyTorch for GPU acceleration"
        })
    
    return {
        "statistics": generation_stats,
        "insights": insights,
        "recommendations": recommendations,
        "current_device": current_device,
        "cuda_available": torch.cuda.is_available()
    }


@app.post("/clear-gpu-cache")
async def clear_gpu_cache():
    """
    Manually clear GPU memory cache.
    
    Returns:
        Cache clearing status and memory info
    """
    if not torch.cuda.is_available():
        return {
            "success": False,
            "message": "CUDA not available"
        }
    
    try:
        # Get memory before clearing
        memory_before = {}
        if current_device and current_device.startswith("cuda"):
            device_id = int(current_device.split(":")[1]) if ":" in current_device else 0
            memory_before = get_cuda_memory_info(device_id)
        
        # Clear cache
        cleanup_gpu_memory()
        
        # Get memory after clearing
        memory_after = {}
        if current_device and current_device.startswith("cuda"):
            memory_after = get_cuda_memory_info(device_id)
        
        return {
            "success": True,
            "message": "GPU cache cleared successfully",
            "memory_before": memory_before,
            "memory_after": memory_after,
            "memory_freed_gb": round(memory_before.get("cached", 0) - memory_after.get("cached", 0), 2)
        }
        
    except Exception as e:
        return {
            "success": False,
            "message": f"Failed to clear GPU cache: {str(e)}"
        }


# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def root():
    """Serve the main web interface"""
    return FileResponse(STATIC_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn
    
    # Run the server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
