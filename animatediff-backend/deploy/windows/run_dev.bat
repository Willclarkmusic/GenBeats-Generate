@echo off
REM Start the AnimateDiff backend development server

echo ========================================
echo AnimateDiff Backend - Development Server
echo ========================================

if not exist venv (
    echo ERROR: Virtual environment not found
    echo Please run setup_venv.bat first
    pause
    exit /b 1
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Check if .env file exists
if not exist .env (
    echo WARNING: .env file not found
    echo Copying .env.example to .env...
    copy .env.example .env
)

REM Create output directory
if not exist outputs mkdir outputs

REM Start the development server
echo.
echo Starting development server...
echo Server will be available at: http://localhost:8000
echo Press Ctrl+C to stop the server
echo.

python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000