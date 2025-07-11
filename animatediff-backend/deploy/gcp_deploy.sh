#!/bin/bash
# Google Cloud Run deployment script for AnimateDiff backend

set -e

# Configuration
PROJECT_ID="${1:-your-project-id}"
REGION="${2:-us-central1}"
SERVICE_NAME="animatediff-backend"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

echo "🚀 Deploying AnimateDiff Backend to Google Cloud Run"
echo "Project: ${PROJECT_ID}"
echo "Region: ${REGION}"
echo "Service: ${SERVICE_NAME}"

# Check if gcloud is installed and authenticated
if ! command -v gcloud &> /dev/null; then
    echo "❌ gcloud CLI is not installed. Please install it first."
    exit 1
fi

if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    echo "❌ Not authenticated with gcloud. Please run 'gcloud auth login'"
    exit 1
fi

# Set the project
echo "📋 Setting project to ${PROJECT_ID}..."
gcloud config set project ${PROJECT_ID}

# Enable required APIs
echo "🔧 Enabling required APIs..."
gcloud services enable \
    cloudbuild.googleapis.com \
    run.googleapis.com \
    containerregistry.googleapis.com \
    compute.googleapis.com

# Create secrets for sensitive data
echo "🔐 Creating secrets..."
if ! gcloud secrets describe animatediff-secrets &>/dev/null; then
    echo "Creating admin password secret..."
    echo -n "$(openssl rand -base64 32)" | gcloud secrets create animatediff-secrets --data-file=-
    
    echo "Creating JWT secret..."
    echo -n "$(openssl rand -base64 64)" | gcloud secrets create jwt-secret --data-file=-
else
    echo "Secrets already exist"
fi

# Build and push Docker image
echo "🐳 Building Docker image..."
docker build -f Dockerfile.cloud -t ${IMAGE_NAME}:latest .

echo "📤 Pushing image to Container Registry..."
docker push ${IMAGE_NAME}:latest

# Deploy to Cloud Run
echo "☁️ Deploying to Cloud Run..."

# Replace PROJECT_ID in service configuration
sed "s/PROJECT_ID/${PROJECT_ID}/g" cloudrun-service.yaml > cloudrun-service-deploy.yaml

# Deploy the service
gcloud run services replace cloudrun-service-deploy.yaml \
    --region=${REGION}

# Get the service URL
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} \
    --region=${REGION} \
    --format="value(status.url)")

echo ""
echo "✅ Deployment completed successfully!"
echo "🌐 Service URL: ${SERVICE_URL}"
echo "🔑 Admin Password: $(gcloud secrets versions access latest --secret=animatediff-secrets)"
echo ""
echo "💰 Cost Optimization Features:"
echo "  - Scales to zero when idle (no charges)"
echo "  - Per-second billing during use"
echo "  - Automatic shutdown after 1 hour"
echo "  - Maximum 3 concurrent instances"
echo ""
echo "📊 Monitor costs at: https://console.cloud.google.com/billing"

# Clean up temporary file
rm -f cloudrun-service-deploy.yaml

echo "🎬 Your AnimateDiff backend is now live!"
echo "Access it at: ${SERVICE_URL}"