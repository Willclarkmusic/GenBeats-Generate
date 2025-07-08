import pytest
from fastapi.testclient import TestClient
from app.main import app
import os

client = TestClient(app)

def test_health_endpoint():
    """Test the health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"

def test_login_endpoint():
    """Test the login endpoint"""
    # Test with wrong password
    response = client.post("/login", json={"password": "wrong"})
    assert response.status_code == 200
    data = response.json()
    assert data["success"] == False
    
    # Test with correct password
    correct_password = os.getenv("ADMIN_PASSWORD", "admin123")
    response = client.post("/login", json={"password": correct_password})
    assert response.status_code == 200
    data = response.json()
    assert data["success"] == True
    assert "token" in data

def test_root_endpoint():
    """Test the root endpoint serves login page"""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]

def test_app_endpoint():
    """Test the app endpoint serves main application"""
    response = client.get("/app")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]

def test_unauthenticated_access():
    """Test that protected endpoints require authentication"""
    response = client.post("/generate-video", json={
        "prompt": "test prompt",
        "steps": 28,
        "guidance_scale": 7.5,
        "width": 768,
        "height": 512,
        "duration": 16,
        "seed": -1
    })
    assert response.status_code == 401

def test_queue_endpoint_requires_auth():
    """Test that queue endpoint requires authentication"""
    response = client.get("/queue")
    assert response.status_code == 401

def test_videos_endpoint_requires_auth():
    """Test that videos endpoint requires authentication"""
    response = client.get("/videos")
    assert response.status_code == 401

class TestWithAuth:
    """Test class with authentication"""
    
    @pytest.fixture
    def auth_headers(self):
        """Get authentication headers"""
        correct_password = os.getenv("ADMIN_PASSWORD", "admin123")
        response = client.post("/login", json={"password": correct_password})
        token = response.json()["token"]
        return {"Authorization": f"Bearer {token}"}
    
    def test_queue_with_auth(self, auth_headers):
        """Test queue endpoint with authentication"""
        response = client.get("/queue", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert "pending" in data
        assert "processing" in data
        assert "completed" in data
    
    def test_videos_with_auth(self, auth_headers):
        """Test videos endpoint with authentication"""
        response = client.get("/videos", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert "videos" in data
        assert "total" in data
        assert "page" in data
        assert "per_page" in data
    
    def test_storage_with_auth(self, auth_headers):
        """Test storage endpoint with authentication"""
        response = client.get("/storage", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert "video_count" in data
        assert "total_size" in data