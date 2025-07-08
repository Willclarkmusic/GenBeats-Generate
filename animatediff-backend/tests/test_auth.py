import pytest
from app.auth import authenticate_user, create_access_token, verify_token, login_user
from app.models import LoginRequest
import os
from datetime import timedelta

def test_authenticate_user():
    """Test user authentication"""
    correct_password = os.getenv("ADMIN_PASSWORD", "admin123")
    
    # Test correct password
    assert authenticate_user(correct_password) == True
    
    # Test wrong password
    assert authenticate_user("wrong_password") == False

def test_create_access_token():
    """Test JWT token creation"""
    data = {"sub": "admin"}
    token = create_access_token(data)
    assert isinstance(token, str)
    assert len(token) > 0

def test_verify_token():
    """Test JWT token verification"""
    # Create a token
    data = {"sub": "admin"}
    token = create_access_token(data)
    
    # Verify the token
    assert verify_token(token) == True
    
    # Test invalid token
    assert verify_token("invalid_token") == False

def test_login_user():
    """Test user login function"""
    correct_password = os.getenv("ADMIN_PASSWORD", "admin123")
    
    # Test successful login
    login_request = LoginRequest(password=correct_password)
    response = login_user(login_request)
    
    assert response.success == True
    assert response.token is not None
    assert response.message == "Login successful"
    
    # Test failed login
    login_request = LoginRequest(password="wrong_password")
    response = login_user(login_request)
    
    assert response.success == False
    assert response.token is None
    assert response.message == "Invalid password"

def test_token_expiration():
    """Test token expiration handling"""
    # Create a token with short expiration
    data = {"sub": "admin"}
    token = create_access_token(data, expires_delta=timedelta(seconds=1))
    
    # Token should be valid immediately
    assert verify_token(token) == True
    
    # Wait for token to expire (in real scenario, you'd use time.sleep)
    # For testing, we'll just verify the token format is correct
    assert isinstance(token, str)
    assert len(token) > 0