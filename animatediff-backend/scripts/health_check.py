#!/usr/bin/env python3
"""
Health check script for AnimateDiff backend.
This script verifies that the application is running correctly.
"""

import sys
import time
import requests
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_health(host="localhost", port=8000, timeout=30):
    """Check if the AnimateDiff backend is healthy"""
    url = f"http://{host}:{port}/health"
    
    logger.info(f"Checking health at {url}")
    
    try:
        response = requests.get(url, timeout=timeout)
        
        if response.status_code == 200:
            health_data = response.json()
            
            logger.info("✓ Backend is responding")
            logger.info(f"  Status: {health_data.get('status', 'unknown')}")
            logger.info(f"  GPU Available: {health_data.get('gpu_available', 'unknown')}")
            logger.info(f"  Model Loaded: {health_data.get('model_loaded', 'unknown')}")
            logger.info(f"  Queue Size: {health_data.get('queue_size', 'unknown')}")
            logger.info(f"  Active Jobs: {health_data.get('active_jobs', 'unknown')}")
            logger.info(f"  Uptime: {health_data.get('uptime', 'unknown'):.2f}s")
            
            return True
        else:
            logger.error(f"✗ Backend returned status code {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        logger.error("✗ Cannot connect to backend")
        return False
    except requests.exceptions.Timeout:
        logger.error("✗ Request timed out")
        return False
    except Exception as e:
        logger.error(f"✗ Unexpected error: {e}")
        return False

def main():
    """Main health check function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Health check for AnimateDiff backend")
    parser.add_argument("--host", default="localhost", help="Backend host")
    parser.add_argument("--port", type=int, default=8000, help="Backend port")
    parser.add_argument("--timeout", type=int, default=30, help="Request timeout")
    
    args = parser.parse_args()
    
    logger.info("Starting health check...")
    
    if check_health(args.host, args.port, args.timeout):
        logger.info("✓ Health check passed")
        return True
    else:
        logger.error("✗ Health check failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)