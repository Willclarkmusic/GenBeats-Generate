import asyncio
import uuid
from typing import Dict, List, Optional
from datetime import datetime
import os
from .models import Job, JobStatus, VideoGenerationRequest
import logging

logger = logging.getLogger(__name__)

class QueueManager:
    def __init__(self, max_queue_size: int = 10, max_concurrent_jobs: int = 3):
        self.max_queue_size = max_queue_size
        self.max_concurrent_jobs = max_concurrent_jobs
        self.jobs: Dict[str, Job] = {}
        self.pending_queue: List[str] = []
        self.processing_jobs: List[str] = []
        self.completed_jobs: List[str] = []
        self.failed_jobs: List[str] = []
        self._lock = asyncio.Lock()
    
    async def add_job(self, request: VideoGenerationRequest) -> str:
        """Add a new job to the queue"""
        async with self._lock:
            if len(self.pending_queue) >= self.max_queue_size:
                raise ValueError("Queue is full")
            
            job_id = str(uuid.uuid4())
            job = Job(job_id, request)
            self.jobs[job_id] = job
            self.pending_queue.append(job_id)
            
            logger.info(f"Job {job_id} added to queue")
            return job_id
    
    async def get_next_job(self) -> Optional[Job]:
        """Get the next job to process"""
        async with self._lock:
            if not self.pending_queue or len(self.processing_jobs) >= self.max_concurrent_jobs:
                return None
            
            job_id = self.pending_queue.pop(0)
            self.processing_jobs.append(job_id)
            job = self.jobs[job_id]
            job.update_status(JobStatus.PROCESSING, 0.0, "Processing started")
            
            logger.info(f"Job {job_id} started processing")
            return job
    
    async def complete_job(self, job_id: str, result_path: str = None, thumbnail_path: str = None, generation_time: float = None):
        """Mark a job as completed"""
        async with self._lock:
            if job_id in self.processing_jobs:
                self.processing_jobs.remove(job_id)
                self.completed_jobs.append(job_id)
                
                if job_id in self.jobs:
                    job = self.jobs[job_id]
                    job.update_status(JobStatus.COMPLETED, 1.0, "Generation completed")
                    job.result_path = result_path
                    job.thumbnail_path = thumbnail_path
                    job.generation_time = generation_time
                    
                    logger.info(f"Job {job_id} completed successfully")
    
    async def fail_job(self, job_id: str, error: str):
        """Mark a job as failed"""
        async with self._lock:
            if job_id in self.processing_jobs:
                self.processing_jobs.remove(job_id)
                self.failed_jobs.append(job_id)
                
                if job_id in self.jobs:
                    job = self.jobs[job_id]
                    job.update_status(JobStatus.FAILED, 0.0, f"Generation failed: {error}")
                    job.error = error
                    
                    logger.error(f"Job {job_id} failed: {error}")
    
    async def update_job_progress(self, job_id: str, progress: float, message: str = None):
        """Update job progress"""
        async with self._lock:
            if job_id in self.jobs:
                job = self.jobs[job_id]
                job.update_status(job.status, progress, message)
    
    async def get_job(self, job_id: str) -> Optional[Job]:
        """Get a job by ID"""
        return self.jobs.get(job_id)
    
    async def get_queue_status(self) -> Dict:
        """Get current queue status"""
        async with self._lock:
            return {
                "pending": len(self.pending_queue),
                "processing": len(self.processing_jobs),
                "completed": len(self.completed_jobs),
                "failed": len(self.failed_jobs),
                "total_jobs": len(self.jobs)
            }
    
    async def get_recent_jobs(self, limit: int = 50) -> List[Job]:
        """Get recent jobs sorted by creation time"""
        jobs = list(self.jobs.values())
        jobs.sort(key=lambda x: x.created_at, reverse=True)
        return jobs[:limit]
    
    async def cleanup_old_jobs(self, max_age_hours: int = 24):
        """Clean up old jobs and their files"""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        jobs_to_remove = []
        
        async with self._lock:
            for job_id, job in self.jobs.items():
                if job.created_at < cutoff_time and job.status in [JobStatus.COMPLETED, JobStatus.FAILED]:
                    jobs_to_remove.append(job_id)
            
            for job_id in jobs_to_remove:
                job = self.jobs[job_id]
                
                # Clean up files
                if job.result_path and os.path.exists(job.result_path):
                    try:
                        os.remove(job.result_path)
                    except Exception as e:
                        logger.warning(f"Failed to remove video file {job.result_path}: {e}")
                
                if job.thumbnail_path and os.path.exists(job.thumbnail_path):
                    try:
                        os.remove(job.thumbnail_path)
                    except Exception as e:
                        logger.warning(f"Failed to remove thumbnail file {job.thumbnail_path}: {e}")
                
                # Remove from tracking lists
                if job_id in self.completed_jobs:
                    self.completed_jobs.remove(job_id)
                if job_id in self.failed_jobs:
                    self.failed_jobs.remove(job_id)
                
                # Remove from jobs dict
                del self.jobs[job_id]
                
                logger.info(f"Cleaned up old job {job_id}")

# Global queue manager instance
queue_manager = QueueManager(
    max_queue_size=int(os.getenv("MAX_QUEUE_SIZE", "10")),
    max_concurrent_jobs=int(os.getenv("MAX_CONCURRENT_JOBS", "3"))
)