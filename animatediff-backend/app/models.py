from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum
import uuid
from datetime import datetime


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class VideoGenerationRequest(BaseModel):
    prompt: str = Field(..., description="Text prompt for video generation")
    negative_prompt: str = Field("", description="Negative prompt to avoid certain elements")
    steps: int = Field(28, ge=1, le=50, description="Number of inference steps")
    guidance_scale: float = Field(7.5, ge=1.0, le=20.0, description="Guidance scale for generation")
    width: int = Field(768, ge=256, le=1024, description="Video width")
    height: int = Field(512, ge=256, le=1024, description="Video height")
    duration: int = Field(16, ge=8, le=80, description="Number of frames (8 frames = 1 second at 8 FPS)")
    seed: int = Field(-1, description="Random seed (-1 for random)")
    motion_scale: float = Field(1.0, ge=0.1, le=2.0, description="Motion intensity scale")


class VideoGenerationResponse(BaseModel):
    job_id: str
    status: JobStatus
    message: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    progress: float = Field(0.0, ge=0.0, le=1.0)
    message: str
    estimated_time_remaining: Optional[int] = None
    created_at: datetime
    updated_at: datetime


class VideoResultResponse(BaseModel):
    job_id: str
    status: JobStatus
    video_url: Optional[str] = None
    thumbnail_url: Optional[str] = None
    metadata: Dict[str, Any] = {}
    created_at: datetime
    generation_time: Optional[float] = None


class LoginRequest(BaseModel):
    password: str


class LoginResponse(BaseModel):
    success: bool
    message: str
    token: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    gpu_available: bool
    model_loaded: bool
    queue_size: int
    active_jobs: int
    uptime: float


class VideoListResponse(BaseModel):
    videos: List[VideoResultResponse]
    total: int
    page: int
    per_page: int


class Job:
    def __init__(self, job_id: str, request: VideoGenerationRequest):
        self.job_id = job_id
        self.request = request
        self.status = JobStatus.PENDING
        self.progress = 0.0
        self.message = "Job queued"
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.result_path: Optional[str] = None
        self.thumbnail_path: Optional[str] = None
        self.error: Optional[str] = None
        self.generation_time: Optional[float] = None
    
    def update_status(self, status: JobStatus, progress: float = None, message: str = None):
        self.status = status
        if progress is not None:
            self.progress = progress
        if message is not None:
            self.message = message
        self.updated_at = datetime.now()
    
    def to_status_response(self) -> JobStatusResponse:
        return JobStatusResponse(
            job_id=self.job_id,
            status=self.status,
            progress=self.progress,
            message=self.message,
            created_at=self.created_at,
            updated_at=self.updated_at
        )
    
    def to_result_response(self) -> VideoResultResponse:
        return VideoResultResponse(
            job_id=self.job_id,
            status=self.status,
            video_url=f"/result/{self.job_id}/video" if self.result_path else None,
            thumbnail_url=f"/result/{self.job_id}/thumbnail" if self.thumbnail_path else None,
            metadata={
                "prompt": self.request.prompt,
                "negative_prompt": self.request.negative_prompt,
                "steps": self.request.steps,
                "guidance_scale": self.request.guidance_scale,
                "width": self.request.width,
                "height": self.request.height,
                "duration": self.request.duration,
                "seed": self.request.seed,
                "motion_scale": self.request.motion_scale
            },
            created_at=self.created_at,
            generation_time=self.generation_time
        )