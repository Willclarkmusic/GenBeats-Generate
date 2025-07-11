# PowerShell script for Google Cloud deployment from Windows

param(
    [Parameter(Mandatory=$true)]
    [string]$ProjectId,
    
    [Parameter(Mandatory=$false)]
    [string]$Region = "us-central1",
    
    [Parameter(Mandatory=$false)]
    [string]$ServiceName = "animatediff-backend"
)

$ErrorActionPreference = "Stop"

Write-Host "🚀 Deploying AnimateDiff Backend to Google Cloud Run" -ForegroundColor Green
Write-Host "Project: $ProjectId" -ForegroundColor Cyan
Write-Host "Region: $Region" -ForegroundColor Cyan
Write-Host "Service: $ServiceName" -ForegroundColor Cyan

# Check prerequisites
Write-Host "`n🔍 Checking prerequisites..." -ForegroundColor Yellow

# Check if gcloud is installed
try {
    $gcloudVersion = gcloud version 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "gcloud not found"
    }
    Write-Host "✓ Google Cloud SDK installed" -ForegroundColor Green
}
catch {
    Write-Host "❌ Google Cloud SDK not found. Please install from:" -ForegroundColor Red
    Write-Host "   https://cloud.google.com/sdk/docs/install" -ForegroundColor Yellow
    exit 1
}

# Check if Docker is running
try {
    docker version | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "Docker not running"
    }
    Write-Host "✓ Docker is running" -ForegroundColor Green
}
catch {
    Write-Host "❌ Docker is not running. Please start Docker Desktop" -ForegroundColor Red
    exit 1
}

# Check authentication
try {
    $account = gcloud auth list --filter=status:ACTIVE --format="value(account)" 2>$null
    if (-not $account) {
        throw "Not authenticated"
    }
    Write-Host "✓ Authenticated as: $account" -ForegroundColor Green
}
catch {
    Write-Host "❌ Not authenticated with Google Cloud" -ForegroundColor Red
    Write-Host "Please run: gcloud auth login" -ForegroundColor Yellow
    exit 1
}

# Set project
Write-Host "`n📋 Setting project..." -ForegroundColor Yellow
gcloud config set project $ProjectId

# Enable APIs
Write-Host "`n🔧 Enabling required APIs..." -ForegroundColor Yellow
$apis = @(
    "cloudbuild.googleapis.com",
    "run.googleapis.com", 
    "containerregistry.googleapis.com",
    "compute.googleapis.com"
)

foreach ($api in $apis) {
    Write-Host "  Enabling $api..." -ForegroundColor Gray
    gcloud services enable $api
}

# Create secrets
Write-Host "`n🔐 Managing secrets..." -ForegroundColor Yellow

# Check if secrets exist
$secretExists = gcloud secrets describe animatediff-secrets 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "  Creating admin password secret..." -ForegroundColor Gray
    $adminPassword = [System.Web.Security.Membership]::GeneratePassword(16, 4)
    $adminPassword | gcloud secrets create animatediff-secrets --data-file=-
    Write-Host "  Generated admin password: $adminPassword" -ForegroundColor Green
} else {
    Write-Host "  Admin password secret already exists" -ForegroundColor Gray
    $adminPassword = gcloud secrets versions access latest --secret=animatediff-secrets
}

$jwtSecretExists = gcloud secrets describe jwt-secret 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "  Creating JWT secret..." -ForegroundColor Gray
    $jwtSecret = [System.Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes([System.Guid]::NewGuid().ToString() + [System.Guid]::NewGuid().ToString()))
    $jwtSecret | gcloud secrets create jwt-secret --data-file=-
} else {
    Write-Host "  JWT secret already exists" -ForegroundColor Gray
}

# Build Docker image
$imageName = "gcr.io/$ProjectId/$ServiceName"
Write-Host "`n🐳 Building Docker image..." -ForegroundColor Yellow
Write-Host "  Image: $imageName" -ForegroundColor Gray

docker build -f Dockerfile.cloud -t "${imageName}:latest" .
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Docker build failed" -ForegroundColor Red
    exit 1
}

# Push to Container Registry
Write-Host "`n📤 Pushing to Container Registry..." -ForegroundColor Yellow
docker push "${imageName}:latest"
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Docker push failed" -ForegroundColor Red
    exit 1
}

# Prepare service configuration
Write-Host "`n☁️ Deploying to Cloud Run..." -ForegroundColor Yellow
$serviceConfig = Get-Content "cloudrun-service.yaml" -Raw
$serviceConfig = $serviceConfig -replace "PROJECT_ID", $ProjectId
$serviceConfig | Out-File -FilePath "cloudrun-service-deploy.yaml" -Encoding UTF8

# Deploy to Cloud Run
gcloud run services replace cloudrun-service-deploy.yaml --region=$Region
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Cloud Run deployment failed" -ForegroundColor Red
    exit 1
}

# Get service URL
$serviceUrl = gcloud run services describe $ServiceName --region=$Region --format="value(status.url)"

# Clean up
Remove-Item "cloudrun-service-deploy.yaml" -Force

# Success message
Write-Host "`n" + "="*60 -ForegroundColor Green
Write-Host "✅ DEPLOYMENT COMPLETED SUCCESSFULLY!" -ForegroundColor Green
Write-Host "="*60 -ForegroundColor Green

Write-Host "`n🌐 Service URL: $serviceUrl" -ForegroundColor Cyan
Write-Host "🔑 Admin Password: $adminPassword" -ForegroundColor Cyan

Write-Host "`n💰 Cost Optimization Features:" -ForegroundColor Yellow
Write-Host "  ✓ Scales to zero when idle (no charges)" -ForegroundColor Green
Write-Host "  ✓ Per-second billing during use" -ForegroundColor Green  
Write-Host "  ✓ Automatic shutdown after 1 hour" -ForegroundColor Green
Write-Host "  ✓ Maximum 3 concurrent instances" -ForegroundColor Green

Write-Host "`n📊 Monitor costs at:" -ForegroundColor Yellow
Write-Host "  https://console.cloud.google.com/billing" -ForegroundColor Cyan

Write-Host "`n🎬 Your AnimateDiff backend is now live!" -ForegroundColor Green
Write-Host "Access it at: $serviceUrl" -ForegroundColor Cyan

Write-Host "`n💡 Next steps:" -ForegroundColor Yellow
Write-Host "  1. Test the deployment by accessing the URL above" -ForegroundColor Gray
Write-Host "  2. Monitor usage in Google Cloud Console" -ForegroundColor Gray
Write-Host "  3. Set up billing alerts if desired" -ForegroundColor Gray