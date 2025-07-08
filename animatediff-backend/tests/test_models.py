import pytest
from datetime import datetime
from app.models import (
    VideoGenerationRequest, Job, JobStatus, LoginRequest,
    VideoGenerationResponse, JobStatusResponse, VideoResultResponse
)

def test_video_generation_request():
    """Test VideoGenerationRequest model"""
    request = VideoGenerationRequest(
        prompt="A beautiful sunset",
        negative_prompt="blurry",
        steps=28,
        guidance_scale=7.5,
        width=768,
        height=512,
        duration=16,
        seed=-1
    )
    
    assert request.prompt == "A beautiful sunset"
    assert request.negative_prompt == "blurry"
    assert request.steps == 28
    assert request.guidance_scale == 7.5
    assert request.width == 768
    assert request.height == 512
    assert request.duration == 16
    assert request.seed == -1

def test_video_generation_request_defaults():
    """Test VideoGenerationRequest with default values"""
    request = VideoGenerationRequest(prompt="Test prompt")
    
    assert request.prompt == "Test prompt"
    assert request.negative_prompt == ""
    assert request.steps == 28
    assert request.guidance_scale == 7.5
    assert request.width == 768
    assert request.height == 512
    assert request.duration == 16
    assert request.seed == -1

def test_video_generation_request_validation():
    """Test VideoGenerationRequest validation"""
    # Test valid values
    request = VideoGenerationRequest(
        prompt="Test",
        steps=25,
        guidance_scale=8.0,
        width=512,
        height=512,
        duration=20
    )
    assert request.steps == 25
    assert request.guidance_scale == 8.0
    
    # Test boundary values
    request = VideoGenerationRequest(
        prompt="Test",
        steps=1,
        guidance_scale=1.0,
        width=256,
        height=256,
        duration=8
    )
    assert request.steps == 1
    assert request.guidance_scale == 1.0

def test_job_model():
    """Test Job model"""
    request = VideoGenerationRequest(prompt="Test prompt")
    job = Job("test-job-id", request)
    
    assert job.job_id == "test-job-id"
    assert job.request == request
    assert job.status == JobStatus.PENDING
    assert job.progress == 0.0
    assert job.message == "Job queued"
    assert isinstance(job.created_at, datetime)
    assert isinstance(job.updated_at, datetime)

def test_job_update_status():
    """Test Job status update"""
    request = VideoGenerationRequest(prompt="Test prompt")
    job = Job("test-job-id", request)
    
    # Update status
    job.update_status(JobStatus.PROCESSING, 0.5, "Processing...")
    
    assert job.status == JobStatus.PROCESSING
    assert job.progress == 0.5
    assert job.message == "Processing..."
    assert job.updated_at > job.created_at

def test_job_to_status_response():
    """Test Job to status response conversion"""
    request = VideoGenerationRequest(prompt="Test prompt")
    job = Job("test-job-id", request)
    job.update_status(JobStatus.PROCESSING, 0.7, "Almost done...")
    
    response = job.to_status_response()
    
    assert isinstance(response, JobStatusResponse)
    assert response.job_id == "test-job-id"
    assert response.status == JobStatus.PROCESSING
    assert response.progress == 0.7
    assert response.message == "Almost done..."

def test_job_to_result_response():
    """Test Job to result response conversion"""
    request = VideoGenerationRequest(prompt="Test prompt")
    job = Job("test-job-id", request)
    job.update_status(JobStatus.COMPLETED, 1.0, "Completed")
    job.result_path = "/path/to/video.mp4"
    job.thumbnail_path = "/path/to/thumb.jpg"
    job.generation_time = 45.5
    
    response = job.to_result_response()
    
    assert isinstance(response, VideoResultResponse)
    assert response.job_id == "test-job-id"
    assert response.status == JobStatus.COMPLETED
    assert response.video_url == "/result/test-job-id/video"
    assert response.thumbnail_url == "/result/test-job-id/thumbnail"
    assert response.generation_time == 45.5
    assert response.metadata["prompt"] == "Test prompt"

def test_login_request():
    """Test LoginRequest model"""
    request = LoginRequest(password="test_password")
    assert request.password == "test_password"

def test_video_generation_response():
    """Test VideoGenerationResponse model"""
    response = VideoGenerationResponse(
        job_id="test-job-id",
        status=JobStatus.PENDING,
        message="Job queued"
    )
    
    assert response.job_id == "test-job-id"
    assert response.status == JobStatus.PENDING
    assert response.message == "Job queued"

def test_job_status_enum():
    """Test JobStatus enum"""
    assert JobStatus.PENDING == "pending"
    assert JobStatus.PROCESSING == "processing"
    assert JobStatus.COMPLETED == "completed"
    assert JobStatus.FAILED == "failed"