# AnimateDiff Backend - Deployment Guide

## 🚀 Quick Start Summary

Your AnimateDiff backend is now ready for deployment! This guide provides step-by-step instructions for getting your video generation system running.

## 📋 What's Included

### ✅ Complete Application Stack
- **FastAPI Backend** - Full REST API with video generation endpoints
- **AnimateDiff Integration** - Text-to-video generation using state-of-the-art models
- **Password-Protected Frontend** - Clean, responsive web interface
- **Authentication System** - JWT-based session management
- **Queue Management** - Background job processing with real-time updates
- **File Management** - Automatic cleanup and storage optimization

### ✅ Development Environment
- **Python Virtual Environment** - Isolated development setup
- **Hot Reload** - Automatic server restart on code changes
- **Model Download Script** - Automated model setup
- **Health Checks** - System monitoring and diagnostics

### ✅ Production Deployment
- **Multi-Stage Dockerfile** - Optimized container builds
- **GPU/CPU Support** - Automatic detection and fallback
- **Docker Compose** - Multi-container orchestration
- **Cloud-Ready** - Compatible with major cloud platforms

### ✅ Windows Integration
- **Batch Scripts** - One-click deployment commands
- **PowerShell Automation** - Advanced deployment options
- **Environment Management** - Automated setup and configuration

## 🛠️ Deployment Options

### Option 1: Quick Development Setup (Recommended)

1. **Setup Virtual Environment**:
   ```batch
   cd animatediff-backend
   deploy\windows\setup_venv.bat
   ```

2. **Download Models** (5-10 minutes):
   ```batch
   deploy\windows\download_models.bat
   ```

3. **Start Development Server**:
   ```batch
   deploy\windows\run_dev.bat
   ```

4. **Access Application**:
   - URL: http://localhost:8000
   - Password: `admin123`

### Option 2: Docker Production Deployment

1. **Build Docker Image**:
   ```batch
   deploy\windows\build.bat
   ```

2. **Run with GPU** (if available):
   ```batch
   deploy\windows\run-gpu.bat
   ```

3. **Or Run with CPU**:
   ```batch
   deploy\windows\run-cpu.bat
   ```

### Option 3: Advanced PowerShell Deployment

```powershell
# Complete development setup with GPU
.\deploy\windows\deploy.ps1 -Mode Development -GPU

# Docker production with custom settings
.\deploy\windows\deploy.ps1 -Mode Docker -GPU -Port 8080 -Password "mypassword"
```

## 🎯 Next Steps

### 1. Initial Setup
- [ ] Choose deployment method above
- [ ] Run the setup commands
- [ ] Verify the application starts successfully
- [ ] Access the web interface

### 2. Configuration
- [ ] Update password in `.env` file
- [ ] Configure GPU settings if needed
- [ ] Adjust storage limits and cleanup settings
- [ ] Set up SSL/HTTPS for production

### 3. Testing
- [ ] Generate your first video
- [ ] Test different prompts and parameters
- [ ] Verify queue system works
- [ ] Check file cleanup functionality

### 4. Production Deployment
- [ ] Set up cloud hosting (RunPod, GCP, AWS)
- [ ] Configure domain and SSL
- [ ] Set up monitoring and logging
- [ ] Implement backup strategy

## 📊 Performance Expectations

### Hardware Requirements
- **Minimum**: 8GB GPU VRAM, 16GB RAM
- **Recommended**: RTX 4080/4090, 32GB RAM
- **CPU Fallback**: 16GB+ RAM (much slower)

### Generation Times
- **GPU (RTX 4090)**: 30-60 seconds per video
- **GPU (RTX 4080)**: 45-90 seconds per video
- **CPU**: 10-30 minutes per video

### Storage Requirements
- **Models**: ~15GB (cached locally)
- **Per Video**: ~5-50MB (depends on resolution/length)
- **Thumbnails**: ~100KB each

## 🔧 Configuration Options

### Environment Variables (.env)
```bash
# Basic Settings
ADMIN_PASSWORD=your_secure_password
PORT=8000
SECRET_KEY=your_jwt_secret_key

# GPU Settings
CUDA_VISIBLE_DEVICES=0
FORCE_CPU=false

# Storage Settings
MAX_STORED_VIDEOS=100
CLEANUP_INTERVAL=3600

# Queue Settings
MAX_QUEUE_SIZE=10
MAX_CONCURRENT_JOBS=3
```

### Generation Parameters
- **Steps**: 1-50 (quality vs speed trade-off)
- **Guidance Scale**: 1-20 (prompt adherence)
- **Resolution**: 256x256 to 1024x1024
- **Duration**: 8-32 frames (1-4 seconds at 8fps)
- **Motion Scale**: 0.1-2.0 (movement intensity)

## 🌐 Cloud Deployment

### Recommended Platforms
1. **RunPod Serverless** - Best for cost optimization
2. **Google Cloud Run** - Automatic scaling
3. **AWS Lambda** - Serverless execution
4. **Azure Container Instances** - Simple deployment

### Docker Registry Example
```bash
# Build and tag
docker build -t animatediff-api .
docker tag animatediff-api gcr.io/your-project/animatediff-api

# Push to registry
docker push gcr.io/your-project/animatediff-api
```

## 🔍 Troubleshooting

### Common Issues

**GPU Not Detected**:
```bash
# Check GPU
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

**Model Download Fails**:
```bash
# Retry download
python scripts/download_models.py
```

**Port Already in Use**:
```bash
# Change port in .env
PORT=8001
```

**Container Won't Start**:
```bash
# Check logs
docker logs animatediff
```

## 📝 Features Overview

### Web Interface
- **Login Page** - Password protection
- **Generation Form** - All parameters configurable
- **Real-time Progress** - Live updates during generation
- **Video Gallery** - Browse and manage generated videos
- **Presets** - Quick parameter configurations

### API Endpoints
- `POST /generate-video` - Start video generation
- `GET /status/{job_id}` - Check generation progress
- `GET /result/{job_id}/video` - Download video
- `GET /videos` - List all videos
- `DELETE /videos/{job_id}` - Delete video

### Management Features
- **Queue System** - Multiple concurrent jobs
- **File Cleanup** - Automatic old file removal
- **Storage Limits** - Configurable storage management
- **Health Monitoring** - System status checks

## 💡 Tips for Success

1. **Start with Development Mode** - Easier debugging and testing
2. **Use GPU if Available** - Dramatically faster generation
3. **Monitor Storage** - Videos can accumulate quickly
4. **Experiment with Parameters** - Different settings for different styles
5. **Set Strong Passwords** - Especially for production deployment

## 📞 Support

If you encounter issues:
1. Check the logs for error messages
2. Review the troubleshooting section
3. Verify your hardware meets requirements
4. Check GitHub issues for known problems

## 🎉 You're Ready!

Your AnimateDiff backend is now fully configured and ready for video generation. Choose your deployment method and start creating amazing videos from text prompts!

**Happy video generating!** 🎬✨