# PowerShell deployment script for AnimateDiff backend
# This script provides advanced deployment options and automation

param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("Development", "Production", "Docker")]
    [string]$Mode,
    
    [Parameter(Mandatory=$false)]
    [switch]$GPU,
    
    [Parameter(Mandatory=$false)]
    [switch]$SkipModels,
    
    [Parameter(Mandatory=$false)]
    [int]$Port = 8000,
    
    [Parameter(Mandatory=$false)]
    [string]$Password = "admin123"
)

# Colors for output
$Red = "Red"
$Green = "Green"
$Yellow = "Yellow"
$Blue = "Blue"

function Write-ColorOutput {
    param([string]$Message, [string]$Color = "White")
    Write-Host $Message -ForegroundColor $Color
}

function Test-Prerequisites {
    Write-ColorOutput "Checking prerequisites..." $Blue
    
    # Check Python
    try {
        $pythonVersion = python --version 2>&1
        if ($LASTEXITCODE -ne 0) {
            throw "Python not found"
        }
        Write-ColorOutput "✓ Python: $pythonVersion" $Green
    }
    catch {
        Write-ColorOutput "✗ Python is not installed or not in PATH" $Red
        return $false
    }
    
    # Check Docker if needed
    if ($Mode -eq "Docker") {
        try {
            $dockerVersion = docker --version 2>&1
            if ($LASTEXITCODE -ne 0) {
                throw "Docker not found"
            }
            Write-ColorOutput "✓ Docker: $dockerVersion" $Green
        }
        catch {
            Write-ColorOutput "✗ Docker is not installed or not running" $Red
            return $false
        }
    }
    
    # Check GPU if requested
    if ($GPU) {
        try {
            $gpuInfo = nvidia-smi --query-gpu=name --format=csv,noheader 2>&1
            if ($LASTEXITCODE -eq 0) {
                Write-ColorOutput "✓ GPU: $gpuInfo" $Green
            } else {
                Write-ColorOutput "⚠ GPU requested but nvidia-smi not found" $Yellow
            }
        }
        catch {
            Write-ColorOutput "⚠ GPU requested but nvidia-smi not available" $Yellow
        }
    }
    
    return $true
}

function Setup-Environment {
    Write-ColorOutput "Setting up environment..." $Blue
    
    # Create .env file if it doesn't exist
    if (-not (Test-Path ".env")) {
        Write-ColorOutput "Creating .env file..." $Yellow
        Copy-Item ".env.example" ".env"
        
        # Update password in .env
        $envContent = Get-Content ".env" -Raw
        $envContent = $envContent -replace "ADMIN_PASSWORD=.*", "ADMIN_PASSWORD=$Password"
        $envContent = $envContent -replace "PORT=.*", "PORT=$Port"
        Set-Content ".env" $envContent
        
        Write-ColorOutput "✓ Environment file created" $Green
    }
    
    # Create directories
    $directories = @("outputs", "models")
    foreach ($dir in $directories) {
        if (-not (Test-Path $dir)) {
            New-Item -ItemType Directory -Path $dir | Out-Null
            Write-ColorOutput "✓ Created directory: $dir" $Green
        }
    }
}

function Setup-VirtualEnvironment {
    Write-ColorOutput "Setting up Python virtual environment..." $Blue
    
    # Remove existing venv if it exists
    if (Test-Path "venv") {
        Write-ColorOutput "Removing existing virtual environment..." $Yellow
        Remove-Item -Path "venv" -Recurse -Force
    }
    
    # Create virtual environment
    python -m venv venv
    if ($LASTEXITCODE -ne 0) {
        Write-ColorOutput "✗ Failed to create virtual environment" $Red
        return $false
    }
    
    # Activate virtual environment
    & "venv\Scripts\Activate.ps1"
    
    # Upgrade pip
    Write-ColorOutput "Upgrading pip..." $Yellow
    python -m pip install --upgrade pip
    
    # Install requirements
    Write-ColorOutput "Installing requirements..." $Yellow
    pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) {
        Write-ColorOutput "✗ Failed to install requirements" $Red
        return $false
    }
    
    # Install dev requirements
    pip install -r requirements-dev.txt
    
    Write-ColorOutput "✓ Virtual environment setup complete" $Green
    return $true
}

function Download-Models {
    Write-ColorOutput "Downloading AnimateDiff models..." $Blue
    
    if ($SkipModels) {
        Write-ColorOutput "⚠ Skipping model download as requested" $Yellow
        return $true
    }
    
    # Activate virtual environment if in development mode
    if ($Mode -eq "Development") {
        & "venv\Scripts\Activate.ps1"
    }
    
    python scripts\download_models.py
    if ($LASTEXITCODE -eq 0) {
        Write-ColorOutput "✓ Models downloaded successfully" $Green
        return $true
    } else {
        Write-ColorOutput "✗ Model download failed" $Red
        return $false
    }
}

function Start-DevelopmentServer {
    Write-ColorOutput "Starting development server..." $Blue
    
    # Activate virtual environment
    & "venv\Scripts\Activate.ps1"
    
    Write-ColorOutput "Development server will be available at: http://localhost:$Port" $Green
    Write-ColorOutput "Password: $Password" $Green
    Write-ColorOutput "Press Ctrl+C to stop the server" $Yellow
    
    # Start server
    python -m uvicorn app.main:app --reload --host 0.0.0.0 --port $Port
}

function Build-DockerImage {
    Write-ColorOutput "Building Docker image..." $Blue
    
    docker build -t animatediff-api .
    if ($LASTEXITCODE -eq 0) {
        Write-ColorOutput "✓ Docker image built successfully" $Green
        return $true
    } else {
        Write-ColorOutput "✗ Docker build failed" $Red
        return $false
    }
}

function Start-DockerContainer {
    Write-ColorOutput "Starting Docker container..." $Blue
    
    # Stop existing container
    docker stop animatediff 2>$null
    docker rm animatediff 2>$null
    
    # Build docker run command
    $dockerArgs = @(
        "run", "-d",
        "--name", "animatediff",
        "-p", "$Port:8000",
        "--env-file", ".env",
        "-v", "${PWD}\outputs:/app/outputs"
    )
    
    if ($GPU) {
        $dockerArgs += @("--gpus", "all")
        Write-ColorOutput "Using GPU acceleration" $Green
    } else {
        $dockerArgs += @("-e", "FORCE_CPU=true")
        Write-ColorOutput "Using CPU-only mode" $Yellow
    }
    
    $dockerArgs += "animatediff-api"
    
    # Start container
    & docker $dockerArgs
    
    if ($LASTEXITCODE -eq 0) {
        Write-ColorOutput "✓ Container started successfully" $Green
        Write-ColorOutput "Web interface: http://localhost:$Port" $Green
        Write-ColorOutput "Password: $Password" $Green
        
        # Wait for container to be ready
        Write-ColorOutput "Waiting for container to be ready..." $Yellow
        Start-Sleep -Seconds 5
        
        # Health check
        python scripts\health_check.py --port $Port
        
        return $true
    } else {
        Write-ColorOutput "✗ Failed to start container" $Red
        return $false
    }
}

function Show-Summary {
    Write-ColorOutput "`n" + "="*60 $Blue
    Write-ColorOutput "DEPLOYMENT SUMMARY" $Blue
    Write-ColorOutput "="*60 $Blue
    
    Write-ColorOutput "Mode: $Mode" $Green
    Write-ColorOutput "GPU: $(if ($GPU) { 'Enabled' } else { 'Disabled' })" $Green
    Write-ColorOutput "Port: $Port" $Green
    Write-ColorOutput "Password: $Password" $Green
    Write-ColorOutput "Web interface: http://localhost:$Port" $Green
    
    if ($Mode -eq "Docker") {
        Write-ColorOutput "`nDocker commands:" $Yellow
        Write-ColorOutput "  View logs: docker logs animatediff" $Yellow
        Write-ColorOutput "  Stop container: docker stop animatediff" $Yellow
        Write-ColorOutput "  Remove container: docker rm animatediff" $Yellow
    }
    
    Write-ColorOutput "`nNext steps:" $Yellow
    Write-ColorOutput "1. Open http://localhost:$Port in your browser" $Yellow
    Write-ColorOutput "2. Login with password: $Password" $Yellow
    Write-ColorOutput "3. Start generating videos!" $Yellow
}

# Main execution
Write-ColorOutput "AnimateDiff Backend Deployment Script" $Blue
Write-ColorOutput "Mode: $Mode" $Green

if (-not (Test-Prerequisites)) {
    exit 1
}

Setup-Environment

switch ($Mode) {
    "Development" {
        if (-not (Setup-VirtualEnvironment)) {
            exit 1
        }
        
        if (-not (Download-Models)) {
            exit 1
        }
        
        Start-DevelopmentServer
    }
    
    "Production" {
        if (-not (Setup-VirtualEnvironment)) {
            exit 1
        }
        
        if (-not (Download-Models)) {
            exit 1
        }
        
        Write-ColorOutput "Production mode setup complete" $Green
        Write-ColorOutput "Use run_dev.bat to start the server" $Yellow
    }
    
    "Docker" {
        if (-not (Build-DockerImage)) {
            exit 1
        }
        
        if (-not (Start-DockerContainer)) {
            exit 1
        }
    }
}

Show-Summary