@echo off
REM Build Docker container for AnimateDiff backend

echo ========================================
echo AnimateDiff Backend - Docker Build
echo ========================================

REM Check if Docker is available
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Docker is not installed or not running
    echo Please install Docker Desktop and make sure it's running
    pause
    exit /b 1
)

echo Docker is available
docker --version

echo.
echo Building Docker container...
echo This may take a while on first build.
echo.

docker build -t animatediff-api .

if %errorlevel% equ 0 (
    echo.
    echo ========================================
    echo Docker build completed successfully!
    echo ========================================
    echo.
    echo To run the container:
    echo   GPU:  deploy\windows\run-gpu.bat
    echo   CPU:  deploy\windows\run-cpu.bat
    echo.
) else (
    echo.
    echo ========================================
    echo Docker build failed!
    echo ========================================
    echo.
    echo Please check the error messages above and try again.
    echo.
)

pause