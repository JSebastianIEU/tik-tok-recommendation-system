#!/bin/bash
# Deploy Python recommender service to Google Cloud Run
#
# Prerequisites:
#   1. Install gcloud CLI: https://cloud.google.com/sdk/docs/install
#   2. Run: gcloud auth login
#   3. Run: gcloud config set project YOUR_PROJECT_ID
#
# Usage:
#   bash scripts/deploy_gcloud.sh

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────
PROJECT_ID="${GCP_PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}"
REGION="${GCP_REGION:-europe-west1}"
SERVICE_NAME="tiktok-recommender"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

echo "=== TikTok Recommender Service — Google Cloud Run Deploy ==="
echo "  Project:  ${PROJECT_ID}"
echo "  Region:   ${REGION}"
echo "  Service:  ${SERVICE_NAME}"
echo "  Image:    ${IMAGE_NAME}"
echo ""

# ── Step 1: Enable required APIs ──────────────────────────────────────────
echo "[1/4] Enabling Cloud Run & Container Registry APIs..."
gcloud services enable run.googleapis.com containerregistry.googleapis.com cloudbuild.googleapis.com --quiet

# ── Step 2: Build container with Cloud Build ──────────────────────────────
echo "[2/4] Building container image with Cloud Build (this uploads source + artifacts ~270MB)..."
gcloud builds submit \
  --tag "${IMAGE_NAME}" \
  --dockerfile Dockerfile.recommender \
  --timeout=1200 \
  --quiet

# ── Step 3: Deploy to Cloud Run ──────────────────────────────────────────
echo "[3/4] Deploying to Cloud Run..."
gcloud run deploy "${SERVICE_NAME}" \
  --image "${IMAGE_NAME}" \
  --region "${REGION}" \
  --platform managed \
  --memory 2Gi \
  --cpu 2 \
  --timeout 60 \
  --concurrency 10 \
  --min-instances 0 \
  --max-instances 3 \
  --port 8081 \
  --allow-unauthenticated \
  --set-env-vars "RECOMMENDER_BUNDLE_DIR=artifacts/recommender/20260412T210030Z-phase2-bootstrap-feedback,PYTHONUNBUFFERED=1" \
  --quiet

# ── Step 4: Get the URL ──────────────────────────────────────────────────
echo "[4/4] Getting service URL..."
SERVICE_URL=$(gcloud run services describe "${SERVICE_NAME}" \
  --region "${REGION}" \
  --format "value(status.url)")

echo ""
echo "=== Deployment complete! ==="
echo "  Service URL: ${SERVICE_URL}"
echo ""
echo "  Test health:  curl ${SERVICE_URL}/v1/health"
echo "  Test recommend: curl -X POST ${SERVICE_URL}/v1/recommendations -H 'Content-Type: application/json' -d '{...}'"
echo ""
echo "  Update frontend .env with:"
echo "    RECOMMENDER_BASE_URL=${SERVICE_URL}"
echo "    RECOMMENDER_ENABLED=true"
