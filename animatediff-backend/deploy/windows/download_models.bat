@echo off
REM Download AnimateDiff models in virtual environment

echo ========================================
echo AnimateDiff Backend - Download Models
echo ========================================

if not exist venv (
    echo ERROR: Virtual environment not found
    echo Please run setup_venv.bat first
    pause
    exit /b 1
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Create models directory
if not exist models mkdir models

echo.
echo Downloading AnimateDiff models...
echo This may take a while depending on your internet connection.
echo.

python scripts\download_models.py

if %errorlevel% equ 0 (
    echo.
    echo ========================================
    echo Models downloaded successfully!
    echo ========================================
    echo.
    echo You can now start the development server with:
    echo   deploy\windows\run_dev.bat
    echo.
) else (
    echo.
    echo ========================================
    echo Model download failed!
    echo ========================================
    echo.
    echo Please check the error messages above and try again.
    echo.
)

pause