@echo off
REM Activate the Python virtual environment for AnimateDiff backend

echo ========================================
echo AnimateDiff Backend - Activate Environment
echo ========================================

if not exist venv (
    echo ERROR: Virtual environment not found
    echo Please run setup_venv.bat first
    pause
    exit /b 1
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Virtual environment activated!
echo.
echo Available commands:
echo   python -m uvicorn app.main:app --reload    # Start development server
echo   python scripts\download_models.py          # Download models
echo   python scripts\health_check.py             # Check server health
echo   pytest tests\                              # Run tests
echo.
echo To exit the virtual environment, type: deactivate
echo.

REM Start a new command prompt with the virtual environment activated
cmd /k