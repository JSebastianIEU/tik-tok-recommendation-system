# Deploy Recommender to Google Cloud Run

## Quick Start (5 minutes)

### 1. Install Google Cloud CLI

Download and run the installer:
https://dl.google.com/dl/cloudsdk/channels/rapid/GoogleCloudSDKInstaller.exe

After install, open a **new terminal** and run:
```bash
gcloud init
# Select your project or create one
# Choose region: europe-west1 (or your preferred region)
```

### 2. Authenticate
```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

### 3. Deploy (one command)

From the repo root directory:
```bash
bash scripts/deploy_gcloud.sh
```

Or manually:
```bash
# Enable APIs
gcloud services enable run.googleapis.com containerregistry.googleapis.com cloudbuild.googleapis.com

# Build & push image using Cloud Build (~5-10 min first time)
gcloud builds submit \
  --tag gcr.io/YOUR_PROJECT_ID/tiktok-recommender \
  --dockerfile Dockerfile.recommender \
  --timeout=1200

# Deploy to Cloud Run
gcloud run deploy tiktok-recommender \
  --image gcr.io/YOUR_PROJECT_ID/tiktok-recommender \
  --region europe-west1 \
  --platform managed \
  --memory 2Gi \
  --cpu 2 \
  --timeout 60 \
  --concurrency 10 \
  --min-instances 0 \
  --max-instances 3 \
  --port 8081 \
  --allow-unauthenticated \
  --set-env-vars "RECOMMENDER_BUNDLE_DIR=artifacts/recommender/20260412T210030Z-phase2-bootstrap-feedback,PYTHONUNBUFFERED=1"
```

### 4. Get the Service URL
```bash
gcloud run services describe tiktok-recommender --region europe-west1 --format "value(status.url)"
# Returns something like: https://tiktok-recommender-abc123-ew.a.run.app
```

### 5. Update Frontend Config

Edit `frontend/.env`:
```env
RECOMMENDER_BASE_URL=https://tiktok-recommender-abc123-ew.a.run.app
RECOMMENDER_ENABLED=true
RECOMMENDER_TIMEOUT_MS=30000
```

### 6. Test
```bash
curl https://tiktok-recommender-abc123-ew.a.run.app/v1/health
# Should return: {"status":"ok"}
```

## Cost Estimate

With $260 credit:
- Cloud Build: ~$0.003/build-minute → ~$0.05 per deploy
- Cloud Run: ~$0.00002400/vCPU-second + $0.00000250/GiB-second
- With min-instances=0: **$0 when idle**, ~$0.10/hour under load
- Container Registry: ~$0.026/GB/month for stored images
- **Total for demo/testing: < $5/month**

## Architecture

```
[React Frontend :5173] → [Node Gateway :5174] → [Cloud Run: Python Recommender]
      (local)                 (local)           (https://...run.app)
```

The Node gateway runs locally and proxies to the Cloud Run service.
For full cloud deployment, the Node gateway can also be deployed to Cloud Run.
