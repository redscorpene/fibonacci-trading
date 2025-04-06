#!/bin/bash

# Set project ID
PROJECT_ID="fibonacci-trading-ai"

# Build the Docker images for autonomous services
echo "Building Docker images for autonomous services..."
docker build -t gcr.io/$PROJECT_ID/market-data-collector:latest -f Dockerfile.autonomous .
docker build -t gcr.io/$PROJECT_ID/autonomous-learning:latest -f Dockerfile.autonomous .

# Push the images to Google Container Registry
echo "Pushing images to GCR..."
docker push gcr.io/$PROJECT_ID/market-data-collector:latest
docker push gcr.io/$PROJECT_ID/autonomous-learning:latest

# Deploy market data collector
echo "Deploying market data collector..."
gcloud run deploy market-data-collector \
    --image gcr.io/$PROJECT_ID/market-data-collector:latest \
    --platform managed \
    --region us-central1 \
    --memory 1Gi \
    --cpu 1 \
    --min-instances 1 \
    --command="python" \
    --args="market_data_collector.py" \
    --update-secrets="/app/key.json=fibonacci-key:latest"

# Deploy autonomous learning system
echo "Deploying autonomous learning system..."
gcloud run deploy autonomous-learning \
    --image gcr.io/$PROJECT_ID/autonomous-learning:latest \
    --platform managed \
    --region us-central1 \
    --memory 2Gi \
    --cpu 1 \
    --min-instances 1 \
    --command="python" \
    --args="autonomous_learning.py" \
    --update-secrets="/app/key.json=fibonacci-key:latest"

echo "Autonomous services deployment complete!"

# Update main trading service to use autonomous learning
echo "Updating main trading service..."
gcloud run services update fibonacci-trading \
    --image us-central1-docker.pkg.dev/$PROJECT_ID/fibonacci-trading-repo/fibonacci-trading:latest \
    --platform managed \
    --region us-central1 \
    --set-env-vars="USE_AUTONOMOUS_LEARNING=1"

echo "All deployments complete!"