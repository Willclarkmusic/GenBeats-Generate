@echo off
REM Run AnimateDiff backend in Docker container with CPU only

echo ========================================
echo AnimateDiff Backend - Docker CPU Deploy
echo ========================================

REM Check if Docker is available
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Docker is not installed or not running
    echo Please install Docker Desktop and make sure it's running
    pause
    exit /b 1
)

REM Check if Docker image exists
docker image inspect animatediff-api >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Docker image 'animatediff-api' not found
    echo Please run build.bat first
    pause
    exit /b 1
)

REM Check if .env file exists
if not exist .env (
    echo WARNING: .env file not found
    echo Copying .env.example to .env...
    copy .env.example .env
)

REM Create output directory
if not exist outputs mkdir outputs

REM Stop existing container if running
docker stop animatediff >nul 2>&1
docker rm animatediff >nul 2>&1

echo.
echo Starting Docker container with CPU only...
echo Container will be available at: http://localhost:8000
echo WARNING: CPU-only mode will be significantly slower than GPU mode
echo.

docker run -d ^
  --name animatediff ^
  -p 8000:8000 ^
  -e FORCE_CPU=true ^
  --env-file .env ^
  -v "%cd%\outputs:/app/outputs" ^
  animatediff-api

if %errorlevel% equ 0 (
    echo.
    echo ========================================
    echo Container started successfully!
    echo ========================================
    echo.
    echo Container name: animatediff
    echo Web interface: http://localhost:8000
    echo Mode: CPU only
    echo.
    echo To view logs: docker logs animatediff
    echo To stop: docker stop animatediff
    echo.
    echo Waiting for container to be ready...
    timeout /t 5 /nobreak >nul
    
    REM Health check
    python scripts\health_check.py
    
) else (
    echo.
    echo ========================================
    echo Failed to start container!
    echo ========================================
    echo.
    echo Please check the error messages above.
    echo.
)

pause