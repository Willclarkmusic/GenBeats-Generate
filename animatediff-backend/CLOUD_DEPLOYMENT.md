# 🚀 Google Cloud Run Deployment Guide

Deploy your AnimateDiff backend to Google Cloud Run with **zero-cost idle time** and GPU acceleration.

## 💰 **Cost Optimization Features**

### **Zero-Cost When Idle**
- ✅ **Scales to zero**: No charges when nobody is using the service
- ✅ **Per-second billing**: Only pay during active video generation
- ✅ **Automatic shutdown**: Service stops after 30 minutes of inactivity
- ✅ **Cold start optimization**: Fast startup when first request comes in

### **Expected Costs**
- **Idle time**: $0.00/hour (scales to zero)
- **Active generation**: ~$2.80/hour (GPU + CPU + Memory)
- **Typical 5-minute video generation**: ~$0.23
- **10-second generation**: ~$0.05

## 🛠️ **Prerequisites**

### **1. Google Cloud Setup**
```bash
# Install Google Cloud SDK
# Windows: Download from https://cloud.google.com/sdk/docs/install

# Authenticate
gcloud auth login

# Create project (or use existing)
gcloud projects create your-project-id --name="AnimateDiff Backend"
gcloud config set project your-project-id

# Enable billing (required for GPU usage)
# Go to: https://console.cloud.google.com/billing
```

### **2. Local Requirements**
- Docker Desktop running
- Google Cloud SDK installed
- Windows PowerShell or Command Prompt

## 🚀 **Quick Deployment**

### **Option 1: PowerShell Script (Recommended)**
```powershell
# Run from project directory
.\deploy\windows\deploy_gcp.ps1 -ProjectId "your-project-id"
```

### **Option 2: Manual Steps**
```bash
# 1. Build and push Docker image
docker build -f Dockerfile.cloud -t gcr.io/your-project-id/animatediff-backend .
docker push gcr.io/your-project-id/animatediff-backend

# 2. Deploy to Cloud Run
gcloud run deploy animatediff-backend \
  --image gcr.io/your-project-id/animatediff-backend \
  --region us-central1 \
  --gpu 1 \
  --gpu-type nvidia-l4 \
  --memory 16Gi \
  --cpu 4 \
  --min-instances 0 \
  --max-instances 3 \
  --timeout 3600 \
  --allow-unauthenticated
```

## ⚙️ **Configuration Options**

### **GPU Options**
```yaml
# NVIDIA L4 (Recommended - Best cost/performance)
gpu-type: nvidia-l4
cost: ~$2.50/hour when active

# NVIDIA T4 (Budget option)
gpu-type: nvidia-t4  
cost: ~$1.40/hour when active
```

### **Region Selection**
```bash
# Recommended regions with GPU availability:
us-central1    # Iowa (lowest cost)
us-east1       # South Carolina
us-west1       # Oregon
europe-west4   # Netherlands
asia-southeast1 # Singapore
```

### **Resource Allocation**
```yaml
# Balanced (Recommended)
memory: 16Gi
cpu: 4
max-instances: 3

# Budget (Slower)
memory: 8Gi
cpu: 2
max-instances: 1

# High Performance
memory: 32Gi
cpu: 8
max-instances: 5
```

## 📊 **Cost Monitoring**

### **Built-in Cost Tracking**
Your deployed service includes cost monitoring endpoints:

```bash
# Get current resource usage and costs
curl "https://your-service-url.run.app/cloud/stats" \
  -H "Authorization: Bearer YOUR_TOKEN"

# Get cost estimate for generation time
curl "https://your-service-url.run.app/cloud/cost-estimate?generation_time=300"
```

### **Google Cloud Billing Alerts**
Set up billing alerts to prevent unexpected charges:

```bash
# Create budget alert
gcloud billing budgets create \
  --billing-account=BILLING_ACCOUNT_ID \
  --display-name="AnimateDiff Budget" \
  --budget-amount=50 \
  --threshold-rule=percent=50,basis=current-spend \
  --threshold-rule=percent=90,basis=current-spend \
  --threshold-rule=percent=100,basis=current-spend
```

## 🎯 **Optimization Tips**

### **Minimize Costs**
1. **Use presets**: Optimized parameters for faster generation
2. **Start with shorter videos**: Test with 2-3 seconds first
3. **Lower resolution for testing**: Use 512x512 for quick tests
4. **Set billing alerts**: Monitor spending in real-time
5. **Use during off-peak**: No pricing difference, but better availability

### **Maximize Performance**
1. **Keep service warm**: Generate videos regularly to avoid cold starts
2. **Batch operations**: Generate multiple videos in one session
3. **Monitor resource usage**: Use `/cloud/stats` endpoint
4. **Choose optimal regions**: us-central1 typically has best GPU availability

## 🔒 **Security Configuration**

### **Environment Variables**
```bash
# Production secrets (auto-generated during deployment)
ADMIN_PASSWORD=generated-secure-password
SECRET_KEY=generated-jwt-secret

# Optional customization
MAX_CONCURRENT_JOBS=1        # Limit concurrent generations
IDLE_SHUTDOWN_MINUTES=30     # Auto-shutdown after inactivity
MAX_STORED_VIDEOS=10         # Limit cloud storage usage
```

### **Access Control**
- Service URL is public but requires password authentication
- All API endpoints require JWT tokens
- Automatic cleanup prevents resource exhaustion
- HTTPS enforced by Cloud Run

## 📈 **Scaling Behavior**

### **Automatic Scaling**
```yaml
Cold Start: 0 → 1 instance (30-60 seconds)
Scale Up: 1 → 3 instances (concurrent requests)
Scale Down: 3 → 0 instances (30 minutes idle)
```

### **Performance Characteristics**
- **Cold start**: 30-60 seconds (model loading)
- **Warm requests**: Immediate response
- **Video generation**: 30 seconds - 10 minutes depending on parameters
- **Shutdown**: Automatic after 30 minutes idle

## 🔧 **Troubleshooting**

### **Common Issues**

**GPU Quota Exceeded**:
```bash
# Check quota usage
gcloud compute regions describe us-central1 \
  --format="value(quotas[].usage,quotas[].limit)"

# Request quota increase
# Go to: https://console.cloud.google.com/iam-admin/quotas
```

**Service Not Starting**:
```bash
# Check logs
gcloud run services logs read animatediff-backend \
  --region=us-central1 \
  --limit=50

# Verify image
gcloud container images list --repository=gcr.io/your-project-id
```

**High Costs**:
```bash
# Check current spending
gcloud billing accounts list
gcloud alpha billing accounts describe BILLING_ACCOUNT_ID

# Review resource usage
curl "https://your-service-url.run.app/cloud/stats"
```

## 🎬 **Success Verification**

After deployment, verify everything works:

### **1. Check Service Status**
```bash
gcloud run services describe animatediff-backend \
  --region=us-central1 \
  --format="value(status.url,status.conditions)"
```

### **2. Test Generation**
1. Open service URL in browser
2. Login with generated password
3. Generate a short test video (2 seconds, 512x512)
4. Monitor costs in Google Cloud Console

### **3. Monitor Performance**
```bash
# Resource usage
curl "https://your-service-url.run.app/cloud/stats"

# Recent logs
gcloud run services logs read animatediff-backend --limit=20
```

## 💡 **Cost Optimization Examples**

### **Development Testing** ($0.05 per test)
- Duration: 2 seconds
- Resolution: 512x512
- Steps: 20
- Expected cost: ~$0.05

### **Production Quality** ($0.25 per video)
- Duration: 5 seconds
- Resolution: 768x512
- Steps: 30
- Expected cost: ~$0.25

### **Maximum Quality** ($0.50 per video)
- Duration: 10 seconds
- Resolution: 1024x1024
- Steps: 35
- Expected cost: ~$0.50

Your AnimateDiff backend is now ready for cost-effective cloud deployment! 🎉