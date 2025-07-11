# MusicGen AI - Music Generation Backend with CUDA GPU Acceleration

A complete FastAPI application that serves AI-powered music generation using Meta's MusicGen model with advanced CUDA GPU support. Generate music from text descriptions with a beautiful web interface and professional performance monitoring.

## 🚀 Features

### Core Features
- 🎵 **Text-to-Music Generation**: Create music from descriptive text prompts
- 🎛️ **Multiple Model Sizes**: Choose between small, medium, and large models
- 🎧 **Web Interface**: Clean, responsive UI for easy music generation
- 📱 **Mobile Friendly**: Works seamlessly on desktop and mobile devices
- 📚 **Music Library**: Browse and manage your generated tracks
- 🔄 **Real-time Status**: Live progress tracking for generation jobs
- 📥 **Easy Downloads**: Download generated music in WAV format

### GPU Acceleration & Performance
- 🚀 **CUDA GPU Support**: Full GPU acceleration for NVIDIA RTX 3080 Ti and compatible cards
- ⚡ **Intelligent Device Selection**: Automatic optimal device detection and selection
- 🔄 **CPU Fallback**: Seamless fallback to CPU if GPU is unavailable or out of memory
- 📊 **Performance Monitoring**: Real-time GPU memory, temperature, and utilization tracking
- 💾 **Memory Management**: Smart GPU memory cleanup and optimization
- 📈 **Performance Analytics**: Detailed generation statistics and performance insights
- 🛠️ **Advanced Controls**: Manual device selection and GPU cache management

### Developer Features
- 🔍 **Comprehensive Diagnostics**: Detailed CUDA compatibility and capability reporting
- 📋 **Multiple Requirements Files**: Optimized for different deployment scenarios
- 🐛 **Enhanced Error Handling**: Intelligent error recovery and detailed logging
- 🔧 **API Endpoints**: Complete REST API with performance monitoring endpoints

## Quick Start

### Prerequisites

- Python 3.8 or higher
- Windows 11 with WSL2 (or Linux/macOS)
- At least 4GB RAM (8GB+ recommended for GPU acceleration)
- **For GPU Acceleration (Recommended):**
  - NVIDIA GPU with CUDA Compute Capability 6.0+ (GTX 1060, RTX series, etc.)
  - 4GB+ VRAM (8GB+ recommended for larger models)
  - NVIDIA Driver 450.80.02+ (Windows) or 450.80.02+ (Linux)
  - CUDA 11.8 or 12.1 support

### Installation

1. **Clone or download the project**
   ```bash
   cd /path/to/your/projects
   # If you have the musicgen-backend folder, navigate to it:
   cd musicgen-backend
   ```

2. **Create a Python virtual environment**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**
   
   **On WSL2/Linux/macOS:**
   ```bash
   source venv/bin/activate
   ```
   
   **On Windows Command Prompt:**
   ```cmd
   venv\\Scripts\\activate
   ```

4. **Install dependencies**

   **Option A: For CUDA GPU Acceleration (Recommended for RTX 3080 Ti)**
   ```bash
   pip install --upgrade pip
   pip install -r requirements-cuda.txt
   ```

   **Option B: For CPU-only operation**
   ```bash
   pip install --upgrade pip
   pip install -r requirements-cpu.txt
   ```

   **Option C: Manual CUDA installation**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   # Then install CUDA-enabled PyTorch (see CUDA Setup section below)
   ```

5. **Run the application**
   ```bash
   python main.py
   ```

6. **Open your browser**
   Navigate to: http://localhost:8000

That's it! The application will automatically download the small model on first use.

## Detailed Setup Guide

### System Requirements

**Minimum Requirements:**
- 4GB RAM
- 2GB free disk space
- Internet connection (for model downloads)

**Recommended for Best Performance:**
- 8GB+ RAM
- NVIDIA GPU with 4GB+ VRAM
- 10GB free disk space (for larger models)
- Fast internet connection

## 🚀 CUDA GPU Setup (For Maximum Performance)

### Why Use GPU Acceleration?

GPU acceleration with CUDA provides **5-10x faster** music generation compared to CPU-only processing:

- **RTX 3080 Ti**: ~15 seconds for 30-second track (vs ~2-3 minutes on CPU)
- **RTX 4090**: ~8 seconds for 30-second track  
- **GTX 1660 Ti**: ~45 seconds for 30-second track

### Prerequisites for CUDA Support

1. **Check your GPU compatibility:**
   ```bash
   nvidia-smi
   ```
   You should see your GPU listed with driver version and memory info.

2. **Verify CUDA Compute Capability** (must be 6.0+):
   - RTX 30/40 series: ✅ 8.6+ (Excellent)
   - RTX 20 series: ✅ 7.5 (Very Good)  
   - GTX 16 series: ✅ 7.5 (Good)
   - GTX 10 series: ✅ 6.1 (Acceptable)

### Windows 11 CUDA Installation

#### Step 1: Install/Update NVIDIA Drivers

1. **Download latest drivers** from [NVIDIA's website](https://www.nvidia.com/Download/index.aspx)
2. **Install with "Clean Installation" option**
3. **Restart your computer**
4. **Verify installation:**
   ```cmd
   nvidia-smi
   ```

#### Step 2: CUDA Toolkit Installation (Optional)

For most users, PyTorch's bundled CUDA is sufficient. For advanced users:

1. **Download CUDA Toolkit 11.8** from [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads)
2. **Install with default settings**
3. **Add to PATH** (usually automatic)

#### Step 3: Install CUDA-enabled PyTorch

**Option A: Use our optimized requirements file (Recommended)**
```bash
# In your activated virtual environment
pip install -r requirements-cuda.txt
```

**Option B: Manual installation**
```bash
# Uninstall CPU version if installed
pip uninstall torch torchvision torchaudio

# Install CUDA 11.8 version (recommended for RTX 3080 Ti)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Alternative: CUDA 12.1 version (if you have latest drivers)
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### Step 4: Verify CUDA Installation

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

Expected output for successful installation:
```
CUDA available: True
GPU name: NVIDIA GeForce RTX 3080 Ti
```

### WSL2 CUDA Setup

#### Prerequisites
1. **Windows 11** with WSL2 enabled
2. **NVIDIA Driver 470.76+** installed on Windows (not in WSL)
3. **WSL2 with Ubuntu 20.04+**

#### Installation Steps

1. **Update WSL2 to latest version:**
   ```cmd
   wsl --update
   ```

2. **In WSL2, install CUDA toolkit:**
   ```bash
   # Remove any existing CUDA installations
   sudo apt-get purge nvidia* libnvidia*

   # Update package list
   sudo apt update

   # Install CUDA toolkit
   wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.0-1_all.deb
   sudo dpkg -i cuda-keyring_1.0-1_all.deb
   sudo apt-get update
   sudo apt-get -y install cuda-toolkit-11-8
   ```

3. **Install PyTorch with CUDA:**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

4. **Verify installation:**
   ```bash
   nvidia-smi  # Should show GPU info
   python -c "import torch; print(torch.cuda.is_available())"
   ```

### Cloud Deployment CUDA Setup

#### Google Colab
```bash
# Colab already has CUDA installed
!pip install -r requirements-cuda.txt
```

#### AWS EC2 with GPU
1. **Launch p3, p4, or g4 instance**
2. **Use Deep Learning AMI** (has CUDA pre-installed)
3. **Install dependencies:**
   ```bash
   pip install -r requirements-cuda.txt
   ```

#### Google Cloud Platform
1. **Create VM with GPU (T4, V100, A100)**
2. **Install NVIDIA drivers:**
   ```bash
   curl https://raw.githubusercontent.com/GoogleCloudPlatform/compute-gpu-installation/main/linux/install_gpu_driver.py --output install_gpu_driver.py
   sudo python3 install_gpu_driver.py
   ```

### Troubleshooting CUDA Issues

#### Common Error: "CUDA out of memory"
```bash
# Clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"

# Or use the web interface: Advanced Settings → Clear GPU Cache
```

#### Error: "No CUDA runtime is found"
```bash
# Reinstall PyTorch with CUDA
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Error: "NVIDIA-SMI has failed"
1. **Update NVIDIA drivers**
2. **Restart computer**
3. **Check Windows Device Manager** for driver issues

#### WSL2 GPU not detected
1. **Ensure Windows NVIDIA driver 470.76+**
2. **Update WSL2:** `wsl --update`
3. **Restart WSL:** `wsl --shutdown` then reopen

### Performance Optimization Tips

#### For RTX 3080 Ti Users:
- **Use small model** for development/testing (loads in ~10 seconds)
- **Use medium model** for balanced quality/speed  
- **Use large model** only for final production (requires 6GB+ VRAM)
- **Enable Advanced Settings** → GPU monitoring for real-time stats

#### Memory Management:
- **Close other GPU applications** (games, browsers with hardware acceleration)
- **Use GPU cache clearing** if you get memory errors
- **Monitor GPU memory** in the performance dashboard

#### Optimal Settings:
- **Duration**: 30 seconds is sweet spot for speed/quality
- **Batch size**: Application automatically optimizes based on available VRAM
- **Mixed precision**: Automatically enabled for RTX 20/30/40 series

### Environment Setup in WSL2

If you're using Windows 11 with WSL2:

1. **Ensure WSL2 is properly configured:**
   ```bash
   wsl --version
   ```

2. **Update your Linux distribution:**
   ```bash
   sudo apt update && sudo apt upgrade
   ```

3. **Install Python and pip if not already installed:**
   ```bash
   sudo apt install python3 python3-pip python3-venv
   ```

## 🎵 Usage Guide

### Basic Music Generation

1. **Start the server** (if not already running):
   ```bash
   python main.py
   ```

2. **Open the web interface**: http://localhost:8000

3. **Enter a music description** in the prompt field:
   - "Upbeat electronic dance music with heavy bass"
   - "Relaxing acoustic guitar melody for meditation"
   - "Jazz piano solo in the style of Bill Evans"

4. **Choose your settings**:
   - **Duration**: 2 seconds to 5 minutes (30s recommended)
   - **Model**: Small (fast), Medium (balanced), Large (best quality)

5. **Advanced Settings** (click ⚙️ Advanced Settings):
   - **Random Seed**: For reproducible results
   - **Device Selection**: Auto, Force CPU, or specific GPU
   - **GPU Memory Monitor**: Real-time memory usage (if GPU available)

6. **Click Generate** and monitor real-time progress!

### Performance Monitoring Dashboard

The application includes a comprehensive performance monitoring system:

#### Accessing Performance Data
1. **Click "📈 Performance"** in the music library section
2. **View real-time metrics** in the header after first generation
3. **Monitor GPU status** in Advanced Settings

#### Performance Metrics Available
- **Device Status**: Current GPU/CPU usage, temperature, memory
- **Generation Statistics**: Total generations, GPU vs CPU usage, average times
- **GPU Memory Chart**: Visual breakdown of allocated, cached, and free memory
- **Performance Insights**: Automatic recommendations and optimization tips

#### Real-time Monitoring
- **GPU Temperature**: Live temperature monitoring (if supported)
- **Memory Usage**: Real-time GPU memory utilization
- **Generation Time**: Performance tracking for each generation
- **Device Performance**: Automatic performance comparisons

### Advanced Device Management

#### Automatic Device Selection
The application automatically selects the optimal device:
1. **Detects available GPUs** and their capabilities
2. **Checks memory availability** and compute capability  
3. **Selects best device** based on model requirements
4. **Falls back to CPU** if GPU is unavailable or out of memory

#### Manual Device Control
- **Force CPU**: Use for compatibility or power saving
- **Select Specific GPU**: Choose exact GPU in multi-GPU systems  
- **GPU Cache Management**: Clear GPU memory manually
- **Real-time Monitoring**: Track performance and resource usage

#### Performance Optimization
- **Smart Memory Management**: Automatic cleanup and optimization
- **Mixed Precision**: Automatic FP16/BF16 for compatible GPUs
- **Batch Optimization**: Automatic batch size tuning
- **Error Recovery**: Intelligent fallback on memory errors

### Model Information

| Model | Size | Speed | Quality | RAM Usage | VRAM Usage |
|-------|------|-------|---------|-----------|------------|
| Small | 300M | Fast | Good | ~2GB | ~1GB |
| Medium | 1.5B | Medium | Better | ~4GB | ~3GB |
| Large | 3.3B | Slow | Best | ~8GB | ~6GB |

### API Usage

The API now includes advanced CUDA device control and performance monitoring:

#### Basic Music Generation
```bash
# Generate music with automatic device selection
curl -X POST "http://localhost:8000/generate-music" \
     -H "Content-Type: application/json" \
     -d '{
       "prompt": "Upbeat electronic dance music",
       "duration": 30,
       "model": "small"
     }'

# Generate music with specific device settings
curl -X POST "http://localhost:8000/generate-music" \
     -H "Content-Type: application/json" \
     -d '{
       "prompt": "Relaxing piano music",
       "duration": 60,
       "model": "medium",
       "force_cpu": false,
       "preferred_device": "cuda:0",
       "seed": 12345
     }'

# Check job status (now includes performance data)
curl "http://localhost:8000/job/{job_id}"

# Download generated music
curl "http://localhost:8000/download/{job_id}" -o music.wav
```

#### GPU and Performance Monitoring
```bash
# Get comprehensive system information
curl "http://localhost:8000/system-info"

# Get real-time GPU status
curl "http://localhost:8000/gpu-status"

# Get performance metrics and insights
curl "http://localhost:8000/performance-metrics"

# Clear GPU cache
curl -X POST "http://localhost:8000/clear-gpu-cache"

# List available models
curl "http://localhost:8000/models"
```

#### Enhanced Response Format
The API now returns detailed performance information:

```json
{
  "job_id": "123e4567-e89b-12d3-a456-426614174000",
  "status": "completed",
  "message": "Music generation completed successfully",
  "file_path": "/path/to/generated_music.wav",
  "duration": 30.0,
  "model_used": "small",
  "device_used": "cuda:0",
  "generation_time": 15.2,
  "gpu_memory_used": 2.1,
  "created_at": "2024-01-01T12:00:00"
}
```

## File Structure

```
musicgen-backend/
├── main.py              # FastAPI server application
├── requirements.txt     # Python dependencies
├── README.md           # This file
├── static/             # Web interface files
│   ├── index.html      # Main UI
│   ├── style.css       # Styling
│   └── script.js       # Frontend JavaScript
├── generated_music/    # Generated audio files (auto-created)
├── models/            # Model cache (auto-created)
└── venv/              # Virtual environment (after setup)
```

## Configuration

### Environment Variables

You can customize the application using environment variables:

```bash
# Set the host and port
export HOST=0.0.0.0
export PORT=8000

# Set model cache directory
export MODEL_CACHE_DIR=/path/to/models

# Enable debug logging
export LOG_LEVEL=DEBUG
```

### Model Pre-downloading

To pre-download models for faster first use:

```python
from transformers import MusicgenForConditionalGeneration

# Download models ahead of time
MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-medium")
MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-large")
```

## 🛠️ Troubleshooting

### CUDA-Specific Issues

**1. "CUDA out of memory" error**
```bash
# Solution A: Clear GPU cache via web interface
# Go to Advanced Settings → Clear GPU Cache

# Solution B: Clear cache programmatically
python -c "import torch; torch.cuda.empty_cache()"

# Solution C: Use smaller model or reduce duration
# In web interface: Switch from Large → Medium → Small model
```

**2. "No CUDA runtime is found" or torch.cuda.is_available() returns False**
```bash
# Check NVIDIA driver
nvidia-smi

# Reinstall CUDA-enabled PyTorch
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify installation
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

**3. "NVIDIA-SMI has failed because this driver is not supported"**
- **Update NVIDIA drivers** to latest version
- **Restart computer** after driver installation
- **Check Device Manager** for driver conflicts
- For WSL2: Update Windows NVIDIA driver (not WSL driver)

**4. GPU detected but poor performance**
```bash
# Check if GPU is actually being used
# Look for "cuda:0" in generation logs or web interface

# Verify optimal device selection
curl http://localhost:8000/system-info

# Check GPU utilization during generation
nvidia-smi -l 1  # Live monitoring
```

**5. WSL2 GPU not detected**
```bash
# Update WSL2
wsl --update

# Restart WSL2
wsl --shutdown
# Then reopen WSL terminal

# Check NVIDIA driver in Windows (must be 470.76+)
# In Windows Command Prompt:
nvidia-smi
```

### General Issues

**6. "ModuleNotFoundError" when starting**
```bash
# Make sure virtual environment is activated
source venv/bin/activate  # Linux/macOS/WSL2
# or
venv\\Scripts\\activate   # Windows

# Reinstall dependencies with correct requirements file
pip install -r requirements-cuda.txt  # For GPU
# or
pip install -r requirements-cpu.txt   # For CPU only
```

**7. Slow generation on CPU (when GPU should be available)**
- **Check device selection** in Advanced Settings
- **Verify CUDA installation** with verification commands above
- **Monitor performance dashboard** to see actual device usage
- **Force GPU** by selecting specific device in Advanced Settings

**8. "Port already in use" error**
```bash
# Change the port
uvicorn main:app --host 0.0.0.0 --port 8001

# Or kill existing process
sudo lsof -t -i tcp:8000 | xargs kill -9
```

**9. Audio doesn't play in browser**
- Check browser console for errors
- Try a different browser (Chrome/Firefox recommended)
- Ensure audio files are being generated in `generated_music/`
- Check if file permissions allow browser access

**10. Performance monitoring not working**
```bash
# Install optional monitoring libraries
pip install psutil GPUtil

# Restart the application
python main.py
```

### Performance Optimization Issues

**11. Generation takes longer than expected**
- **Check GPU utilization** in performance dashboard
- **Close other GPU applications** (games, browsers with hardware acceleration)
- **Verify optimal model size** for your GPU memory
- **Use performance insights** in the monitoring dashboard

**12. Memory usage keeps growing**
- **Use GPU cache clearing** after each generation
- **Enable automatic cleanup** (enabled by default)
- **Monitor memory in real-time** via performance dashboard
- **Restart application** if memory leaks persist

### Error Recovery

**13. Application becomes unresponsive**
```bash
# Graceful restart
Ctrl+C  # Stop the server
python main.py  # Restart

# Force kill if needed
ps aux | grep python
kill -9 [process_id]
```

**14. Models fail to download**
- **Check internet connection**
- **Verify disk space** (models are 1-4GB each)
- **Try manual model download**:
```python
from transformers import MusicgenForConditionalGeneration
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
```

### Getting Help

If problems persist:
1. **Check application logs** for detailed error messages
2. **Use performance monitoring** to identify bottlenecks
3. **Try CPU-only mode** to isolate GPU issues
4. **Update all dependencies** to latest versions
5. **Report issues** with system info from `/system-info` endpoint

### Performance Optimization

**For better performance:**

1. **Use GPU acceleration** (install CUDA support)
2. **Increase system RAM** if running out of memory
3. **Use SSD storage** for faster model loading
4. **Close unnecessary applications** to free up resources

### Debug Mode

Run with debug logging for troubleshooting:

```bash
# Enable debug logs
export LOG_LEVEL=DEBUG
python main.py
```

### Getting Help

If you encounter issues:

1. Check the console output for error messages
2. Verify all dependencies are installed correctly
3. Try with the small model first
4. Check available disk space and memory
5. Restart the application

## API Documentation

Once the server is running, visit http://localhost:8000/docs for interactive API documentation.

### Key Endpoints

- `POST /generate-music` - Start music generation
- `GET /job/{job_id}` - Check generation status  
- `GET /download/{job_id}` - Download generated music
- `GET /list-generated` - List all generated files
- `GET /system-info` - Get system information
- `GET /models` - List available models

## Development

### Running in Development Mode

```bash
# Install development dependencies
pip install fastapi uvicorn python-multipart

# Run with auto-reload
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Custom Model Integration

To add support for other models, modify the `AVAILABLE_MODELS` dictionary in `main.py`:

```python
AVAILABLE_MODELS = {
    "small": "facebook/musicgen-small",
    "medium": "facebook/musicgen-medium", 
    "large": "facebook/musicgen-large",
    "custom": "path/to/your/model"
}
```

## License

This project uses Meta's MusicGen models, which have their own licensing terms. Please review the model licenses before commercial use.

## Contributing

Feel free to submit issues and enhancement requests!

---

**Enjoy creating music with AI!** 🎵