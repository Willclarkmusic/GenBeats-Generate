@echo off
REM Setup Python virtual environment for AnimateDiff backend
REM This script creates a virtual environment and installs all dependencies

echo ========================================
echo AnimateDiff Backend - Virtual Environment Setup
echo ========================================

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.10+ and add it to PATH
    pause
    exit /b 1
)

REM Get Python version
for /f "tokens=2" %%i in ('python --version') do set PYTHON_VERSION=%%i
echo Python version: %PYTHON_VERSION%

REM Create virtual environment
echo.
echo Creating virtual environment...
if exist venv (
    echo Virtual environment already exists. Removing old one...
    rmdir /s /q venv
)

python -m venv venv
if %errorlevel% neq 0 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)

echo Virtual environment created successfully.

REM Activate virtual environment
echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo.
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install dependencies
echo.
echo Installing dependencies...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo Installing development dependencies...
pip install -r requirements-dev.txt
if %errorlevel% neq 0 (
    echo WARNING: Failed to install development dependencies
)

REM Verify installation
echo.
echo Verifying installation...
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import diffusers; print(f'Diffusers version: {diffusers.__version__}')"
python -c "import fastapi; print(f'FastAPI version: {fastapi.__version__}')"

echo.
echo ========================================
echo Virtual environment setup complete!
echo ========================================
echo.
echo To activate the virtual environment, run:
echo   venv\Scripts\activate.bat
echo.
echo To download models, run:
echo   python scripts\download_models.py
echo.
echo To start the development server, run:
echo   python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
echo.
echo Or use the convenience scripts:
echo   deploy\windows\activate_env.bat
echo   deploy\windows\run_dev.bat
echo.
pause