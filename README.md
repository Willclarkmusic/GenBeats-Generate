# AnimateDiff Backend

A FastAPI backend for generating videos using AnimateDiff with both virtual environment development and Docker deployment support.

## Features

- 🎥 **AnimateDiff Integration**: Generate high-quality 10-20 second videos from text prompts
- 🔐 **Password Protected**: Simple authentication system with session management
- 💻 **Dual Environment**: Virtual environment for development, Docker for production
- 🎛️ **Parameter Control**: Full control over generation parameters (steps, guidance, resolution, etc.)
- 📱 **Modern Web UI**: Clean, responsive interface with real-time progress tracking
- 🚀 **GPU Acceleration**: CUDA support with automatic CPU fallback
- 🔄 **Queue Management**: Background job processing with status tracking
- 📁 **File Management**: Automatic cleanup and storage limits
- 🐳 **Docker Ready**: Multi-stage Dockerfile with GPU/CPU support

## Quick Start

### Option 1: Windows Virtual Environment (Recommended for Development)

1. **Setup Virtual Environment**:
   ```batch
   deploy\windows\setup_venv.bat
   ```

2. **Download Models**:
   ```batch
   deploy\windows\download_models.bat
   ```

3. **Start Development Server**:
   ```batch
   deploy\windows\run_dev.bat
   ```

4. **Access the Application**:
   - Open http://localhost:8000
   - Default password: `admin123`

### Option 2: Docker Deployment

1. **Build Docker Image**:
   ```batch
   deploy\windows\build.bat
   ```

2. **Run with GPU**:
   ```batch
   deploy\windows\run-gpu.bat
   ```

   **Or Run with CPU**:
   ```batch
   deploy\windows\run-cpu.bat
   ```

### Option 3: Advanced PowerShell Deployment

```powershell
# Development mode
.\deploy\windows\deploy.ps1 -Mode Development -GPU

# Docker mode with GPU
.\deploy\windows\deploy.ps1 -Mode Docker -GPU -Port 8000 -Password "mypassword"

# Production setup
.\deploy\windows\deploy.ps1 -Mode Production -GPU -SkipModels
```

## Project Structure

```
animatediff-backend/
├── app/                        # FastAPI application
│   ├── main.py                 # Main application entry point
│   ├── models.py               # Pydantic models
│   ├── auth.py                 # Authentication system
│   ├── animatediff_handler.py  # AnimateDiff integration
│   ├── queue_manager.py        # Job queue management
│   ├── file_manager.py         # File storage management
│   └── static/                 # Frontend files
│       ├── index.html          # Main application
│       ├── login.html          # Login page
│       ├── script.js           # JavaScript functionality
│       └── style.css           # Styling
├── scripts/                    # Utility scripts
│   ├── download_models.py      # Model download script
│   └── health_check.py         # Health check script
├── deploy/windows/             # Windows deployment scripts
│   ├── setup_venv.bat          # Virtual environment setup
│   ├── activate_env.bat        # Activate environment
│   ├── run_dev.bat             # Development server
│   ├── download_models.bat     # Model download
│   ├── build.bat               # Docker build
│   ├── run-gpu.bat             # GPU container
│   ├── run-cpu.bat             # CPU container
│   └── deploy.ps1              # PowerShell deployment
├── tests/                      # Test files
├── outputs/                    # Generated video storage
├── models/                     # Model cache directory
├── requirements.txt            # Python dependencies
├── requirements-dev.txt        # Development dependencies
├── Dockerfile                  # Multi-stage Docker build
├── docker-compose.yml          # Docker Compose configuration
└── .env.example                # Environment variables template
```

## Development Workflow

### Virtual Environment Setup

1. **Create Environment**:
   ```batch
   python -m venv venv
   venv\Scripts\activate
   ```

2. **Install Dependencies**:
   ```batch
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

3. **Download Models**:
   ```batch
   python scripts\download_models.py
   ```

4. **Start Development Server**:
   ```batch
   python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

### Docker Development

1. **Build Development Image**:
   ```batch
   docker build -t animatediff-api .
   ```

2. **Run Development Container**:
   ```batch
   docker-compose up animatediff-dev
   ```

3. **Run Production Container**:
   ```batch
   docker-compose up animatediff-prod
   ```

## API Endpoints

### Authentication
- `POST /login` - User authentication
- `GET /` - Login page
- `GET /app` - Main application (protected)

### Video Generation
- `POST /generate-video` - Start video generation
- `GET /status/{job_id}` - Check job status
- `GET /result/{job_id}` - Get job result
- `GET /result/{job_id}/video` - Download video file
- `GET /result/{job_id}/thumbnail` - Download thumbnail

### Management
- `GET /health` - System health check
- `GET /videos` - List generated videos
- `DELETE /videos/{job_id}` - Delete video
- `GET /queue` - Queue information
- `GET /storage` - Storage information
- `POST /cleanup` - Manual cleanup

## Configuration

### Environment Variables

```bash
# GPU Configuration
CUDA_VISIBLE_DEVICES=0          # GPU device ID
FORCE_CPU=false                 # Force CPU-only mode

# Model Cache
TORCH_HOME=./models             # PyTorch model cache
HF_HOME=./models                # Hugging Face model cache
MODEL_CACHE_DIR=./models        # Model cache directory

# Application Settings
OUTPUT_DIR=./outputs            # Video output directory
PORT=8000                       # Server port
HOST=0.0.0.0                    # Server host
ADMIN_PASSWORD=admin123         # Frontend password
SECRET_KEY=your_secret_key      # JWT secret key

# Queue Management
MAX_QUEUE_SIZE=10               # Maximum queued jobs
MAX_CONCURRENT_JOBS=3           # Concurrent processing jobs
MAX_STORED_VIDEOS=100           # Maximum stored videos
CLEANUP_INTERVAL=3600           # Cleanup interval (seconds)

# Timeouts
MODEL_DOWNLOAD_TIMEOUT=600      # Model download timeout
VIDEO_GENERATION_TIMEOUT=300    # Video generation timeout
SESSION_EXPIRE_MINUTES=1440     # Session expiration
```

### Generation Parameters

```python
{
    "prompt": "A beautiful sunset over mountains",
    "negative_prompt": "blurry, low quality",
    "steps": 28,                # 1-50 inference steps
    "guidance_scale": 7.5,      # 1.0-20.0 guidance scale
    "width": 768,               # 256-1024 video width
    "height": 512,              # 256-1024 video height
    "duration": 16,             # 8-32 frames
    "motion_scale": 1.0,        # 0.1-2.0 motion intensity
    "seed": -1                  # Random seed (-1 for random)
}
```

## GPU Requirements

### Minimum Requirements
- NVIDIA GPU with 8GB VRAM
- CUDA 11.8 or later
- GPU compute capability 6.0+

### Recommended Requirements
- NVIDIA RTX 4080/4090 or better
- 16GB+ VRAM for higher resolutions
- Fast NVMe SSD for model storage

### CPU Fallback
- 16GB+ RAM recommended
- Modern multi-core CPU
- Generation will be significantly slower

## Cloud Deployment

### Recommended Platforms

1. **RunPod Serverless** (Best for cost optimization)
   - Scale to zero billing
   - RTX 4090: ~$0.40/hour
   - A100 80GB: ~$2.17/hour

2. **Google Cloud Run with GPU**
   - Automatic scaling
   - NVIDIA L4 GPUs
   - Pay-per-use pricing

3. **AWS Lambda with GPU**
   - Serverless execution
   - 15-minute timeout limit

### Docker Registry Push

```bash
# Tag for cloud deployment
docker tag animatediff-api gcr.io/your-project/animatediff-api

# Push to registry
docker push gcr.io/your-project/animatediff-api
```

## Performance Optimization

### GPU Optimization
- Model caching reduces cold start time
- VAE slicing for memory efficiency
- Model CPU offloading when needed

### Memory Management
- Automatic cleanup of old videos
- Storage limits enforcement
- Queue size limitations

### Network Optimization
- Compressed video output
- Thumbnail generation
- Progressive loading

## Troubleshooting

### Common Issues

1. **GPU Not Detected**:
   ```bash
   # Check GPU availability
   nvidia-smi
   python -c "import torch; print(torch.cuda.is_available())"
   ```

2. **Model Download Fails**:
   ```bash
   # Retry model download
   python scripts/download_models.py
   ```

3. **Out of Memory**:
   ```bash
   # Use CPU mode
   set FORCE_CPU=true
   ```

4. **Port Already in Use**:
   ```bash
   # Change port in .env
   PORT=8001
   ```

### Docker Issues

1. **GPU Not Available in Container**:
   ```bash
   # Check Docker GPU support
   docker run --rm --gpus all nvidia/cuda:11.8-base nvidia-smi
   ```

2. **Container Won't Start**:
   ```bash
   # Check container logs
   docker logs animatediff
   ```

3. **Permission Issues**:
   ```bash
   # Fix output directory permissions
   chmod 755 outputs/
   ```

## Development

### Running Tests

```bash
# Activate virtual environment
venv\Scripts\activate

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=app --cov-report=html
```

### Code Quality

```bash
# Format code
black app/ scripts/

# Lint code
flake8 app/ scripts/

# Type checking
mypy app/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions:
- Check the troubleshooting section
- Review the logs for error messages
- Open an issue on GitHub

## Changelog

### v1.0.0
- Initial release
- AnimateDiff integration
- Virtual environment support
- Docker deployment
- Windows deployment scripts
- Password-protected frontend
- Queue management system
- File storage management