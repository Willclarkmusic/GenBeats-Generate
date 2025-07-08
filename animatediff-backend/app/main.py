from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Optional, List
import uvicorn

from .models import (
    VideoGenerationRequest, VideoGenerationResponse, JobStatusResponse,
    VideoResultResponse, LoginRequest, LoginResponse, HealthResponse,
    VideoListResponse, JobStatus
)
from .auth import get_current_user, get_current_user_optional, login_user
from .queue_manager import queue_manager
from .animatediff_handler import animatediff_handler
from .file_manager import file_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="AnimateDiff Video Generation API",
    description="A FastAPI backend for generating videos using AnimateDiff",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Global variables
app_start_time = time.time()
background_task_running = False

# Background task for processing jobs
async def process_jobs():
    """Background task to process jobs from the queue"""
    global background_task_running
    background_task_running = True
    
    logger.info("Background job processor started")
    
    while background_task_running:
        try:
            # Get next job from queue
            job = await queue_manager.get_next_job()
            
            if job is None:
                # No jobs to process, wait a bit
                await asyncio.sleep(1)
                continue
            
            logger.info(f"Processing job {job.job_id}")
            
            try:
                # Generate video
                result = await animatediff_handler.generate_video(job)
                
                # Complete job
                await queue_manager.complete_job(
                    job.job_id,
                    result["video_path"],
                    result["thumbnail_path"],
                    result["generation_time"]
                )
                
                logger.info(f"Job {job.job_id} completed successfully")
                
            except Exception as e:
                logger.error(f"Error processing job {job.job_id}: {e}")
                await queue_manager.fail_job(job.job_id, str(e))
                
        except Exception as e:
            logger.error(f"Error in background job processor: {e}")
            await asyncio.sleep(5)  # Wait before retrying

# Background task for cleanup
async def cleanup_task():
    """Background task for periodic cleanup"""
    while True:
        try:
            await asyncio.sleep(3600)  # Run every hour
            
            # Clean up old jobs
            await queue_manager.cleanup_old_jobs(max_age_hours=24)
            
            # Clean up old files
            await file_manager.cleanup_old_files(max_age_hours=24)
            
            # Enforce storage limit
            await file_manager.enforce_storage_limit()
            
            logger.info("Cleanup task completed")
            
        except Exception as e:
            logger.error(f"Error in cleanup task: {e}")

@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    logger.info("Starting AnimateDiff API server")
    
    # Start background tasks
    asyncio.create_task(process_jobs())
    asyncio.create_task(cleanup_task())
    
    # Load model in background
    asyncio.create_task(animatediff_handler.load_model())
    
    logger.info("AnimateDiff API server started")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources"""
    global background_task_running
    background_task_running = False
    
    logger.info("Shutting down AnimateDiff API server")
    
    # Clean up handler
    animatediff_handler.cleanup()

# Routes

@app.get("/", response_class=HTMLResponse)
async def serve_login():
    """Serve the login page"""
    login_file = static_dir / "login.html"
    if login_file.exists():
        return FileResponse(login_file, media_type="text/html")
    else:
        return HTMLResponse("""
        <!DOCTYPE html>
        <html>
        <head><title>Login</title></head>
        <body>
        <h1>Login</h1>
        <form id="loginForm">
            <input type="password" id="password" placeholder="Password" required>
            <button type="submit">Login</button>
        </form>
        <script>
        document.getElementById('loginForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const password = document.getElementById('password').value;
            const response = await fetch('/login', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({password})
            });
            const result = await response.json();
            if (result.success) {
                localStorage.setItem('token', result.token);
                window.location.href = '/app';
            } else {
                alert(result.message);
            }
        });
        </script>
        </body>
        </html>
        """)

@app.get("/app", response_class=HTMLResponse)
async def serve_app():
    """Serve the main application (protected)"""
    app_file = static_dir / "index.html"
    if app_file.exists():
        return FileResponse(app_file, media_type="text/html")
    else:
        return HTMLResponse("""
        <!DOCTYPE html>
        <html>
        <head><title>AnimateDiff Video Generator</title></head>
        <body>
        <h1>AnimateDiff Video Generator</h1>
        <p>Main application would be here</p>
        </body>
        </html>
        """)

@app.post("/login", response_model=LoginResponse)
async def login(login_request: LoginRequest):
    """User login endpoint"""
    return login_user(login_request)

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    model_info = animatediff_handler.get_model_info()
    queue_status = await queue_manager.get_queue_status()
    
    return HealthResponse(
        status="healthy",
        gpu_available=model_info["gpu_available"],
        model_loaded=model_info["model_loaded"],
        queue_size=queue_status["pending"],
        active_jobs=queue_status["processing"],
        uptime=time.time() - app_start_time
    )

@app.post("/generate-video", response_model=VideoGenerationResponse)
async def generate_video(
    request: VideoGenerationRequest,
    current_user: dict = Depends(get_current_user)
):
    """Generate a video from text prompt"""
    try:
        job_id = await queue_manager.add_job(request)
        
        return VideoGenerationResponse(
            job_id=job_id,
            status=JobStatus.PENDING,
            message="Job queued for processing"
        )
        
    except ValueError as e:
        raise HTTPException(status_code=429, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating video generation job: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/status/{job_id}", response_model=JobStatusResponse)
async def get_job_status(
    job_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get the status of a video generation job"""
    job = await queue_manager.get_job(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return job.to_status_response()

@app.get("/result/{job_id}", response_model=VideoResultResponse)
async def get_job_result(
    job_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get the result of a video generation job"""
    job = await queue_manager.get_job(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return job.to_result_response()

@app.get("/result/{job_id}/video")
async def get_video_file(
    job_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Download the generated video file"""
    job = await queue_manager.get_job(job_id)
    
    if not job or job.status != JobStatus.COMPLETED:
        raise HTTPException(status_code=404, detail="Video not found or not ready")
    
    if not job.result_path or not file_manager.file_exists(job.result_path):
        raise HTTPException(status_code=404, detail="Video file not found")
    
    return FileResponse(
        job.result_path,
        media_type="video/mp4",
        filename=f"{job_id}.mp4"
    )

@app.get("/result/{job_id}/thumbnail")
async def get_thumbnail_file(
    job_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Download the generated thumbnail file"""
    job = await queue_manager.get_job(job_id)
    
    if not job or job.status != JobStatus.COMPLETED:
        raise HTTPException(status_code=404, detail="Thumbnail not found or not ready")
    
    if not job.thumbnail_path or not file_manager.file_exists(job.thumbnail_path):
        raise HTTPException(status_code=404, detail="Thumbnail file not found")
    
    return FileResponse(
        job.thumbnail_path,
        media_type="image/jpeg",
        filename=f"{job_id}_thumb.jpg"
    )

@app.get("/videos", response_model=VideoListResponse)
async def list_videos(
    page: int = 1,
    per_page: int = 20,
    current_user: dict = Depends(get_current_user)
):
    """List all generated videos"""
    try:
        # Get recent jobs
        jobs = await queue_manager.get_recent_jobs(limit=per_page * 10)
        
        # Filter completed jobs
        completed_jobs = [job for job in jobs if job.status == JobStatus.COMPLETED]
        
        # Paginate
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        page_jobs = completed_jobs[start_idx:end_idx]
        
        # Convert to response format
        videos = [job.to_result_response() for job in page_jobs]
        
        return VideoListResponse(
            videos=videos,
            total=len(completed_jobs),
            page=page,
            per_page=per_page
        )
        
    except Exception as e:
        logger.error(f"Error listing videos: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.delete("/videos/{job_id}")
async def delete_video(
    job_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Delete a generated video"""
    try:
        # Delete files
        await file_manager.delete_video_files(job_id)
        
        return {"message": "Video deleted successfully"}
        
    except Exception as e:
        logger.error(f"Error deleting video {job_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/queue")
async def get_queue_info(current_user: dict = Depends(get_current_user)):
    """Get queue information"""
    return await queue_manager.get_queue_status()

@app.get("/storage")
async def get_storage_info(current_user: dict = Depends(get_current_user)):
    """Get storage information"""
    return file_manager.get_storage_info()

@app.post("/cleanup")
async def manual_cleanup(current_user: dict = Depends(get_current_user)):
    """Manually trigger cleanup"""
    try:
        await queue_manager.cleanup_old_jobs(max_age_hours=1)
        await file_manager.cleanup_old_files(max_age_hours=1)
        await file_manager.enforce_storage_limit()
        
        return {"message": "Cleanup completed successfully"}
        
    except Exception as e:
        logger.error(f"Error during manual cleanup: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )