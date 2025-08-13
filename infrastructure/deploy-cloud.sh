#!/bin/bash

# FusionAlpha Cloud Deployment Script
# Supports Railway, DigitalOcean, and Heroku

set -e

echo "FusionAlpha Cloud Deployment"
echo "============================"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker is required but not installed."
    echo "Please install Docker: https://docs.docker.com/get-docker/"
    exit 1
fi

# Function to deploy to Railway
deploy_railway() {
    echo "Deploying to Railway..."
    
    if ! command -v railway &> /dev/null; then
        echo "Installing Railway CLI..."
        npm install -g @railway/cli
    fi
    
    echo "Logging in to Railway..."
    railway login
    
    echo "Creating new Railway project..."
    railway create fusionalpha-pipeline
    
    echo "Deploying..."
    railway up
    
    echo "SUCCESS: Deployed to Railway!"
    echo "Your app will be available at: https://fusionalpha-pipeline.railway.app"
}

# Function to deploy to DigitalOcean
deploy_digitalocean() {
    echo "Deploying to DigitalOcean App Platform..."
    
    if ! command -v doctl &> /dev/null; then
        echo "Installing DigitalOcean CLI..."
        cd ~
        wget https://github.com/digitalocean/doctl/releases/download/v1.100.0/doctl-1.100.0-linux-amd64.tar.gz
        tar xf ~/doctl-1.100.0-linux-amd64.tar.gz
        sudo mv ~/doctl /usr/local/bin
    fi
    
    echo "Please authenticate with DigitalOcean:"
    doctl auth init
    
    # Create app spec
    mkdir -p .do
    cat > .do/app.yaml << EOF
name: fusionalpha-pipeline
services:
- name: web
  source_dir: /
  github:
    repo: rochmanofenna/FusionAlpha
    branch: main
  run_command: python3 main.py --mode web
  environment_slug: python
  instance_count: 1
  instance_size_slug: professional-s
  http_port: 5000
  env:
  - key: USE_CUDA
    value: "false"
  - key: FLASK_ENV
    value: "production"
EOF
    
    echo "Creating DigitalOcean app..."
    doctl apps create .do/app.yaml
    
    echo "SUCCESS: Deployed to DigitalOcean!"
    echo "Check your apps at: https://cloud.digitalocean.com/apps"
}

# Function to deploy to AWS (using existing infrastructure)
deploy_aws() {
    echo "Deploying to AWS using existing Terraform infrastructure..."
    
    # Check if Terraform is installed
    if ! command -v terraform &> /dev/null; then
        echo "ERROR: Terraform is required for AWS deployment."
        echo "Please install Terraform or use Railway/DigitalOcean instead."
        exit 1
    fi
    
    cd ../infrastructure/terraform/gpu-cluster
    
    echo "Initializing Terraform..."
    terraform init
    
    echo "Planning deployment..."
    terraform plan -var="environment=production"
    
    echo "WARNING: This will create EXPENSIVE GPU instances!"
    echo "Continue? (y/N)"
    read -r response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        terraform apply -auto-approve
        echo "SUCCESS: Deployed to AWS!"
    else
        echo "Deployment cancelled."
    fi
}

# Function to build and test locally
test_local() {
    echo "Testing deployment locally..."
    
    echo "Building Docker image..."
    docker build -t fusionalpha-pipeline .
    
    echo "Running container..."
    docker run -p 5000:5000 --name fusionalpha-test fusionalpha-pipeline
    
    echo "SUCCESS: Test deployment running at http://localhost:5000"
}

# Main menu
echo "Choose deployment option:"
echo "1) Railway (Recommended for quick deploy)"
echo "2) DigitalOcean App Platform"
echo "3) AWS (Expensive GPU instances)"
echo "4) Test locally with Docker"
echo "5) Exit"

read -p "Enter your choice [1-5]: " choice

case $choice in
    1)
        deploy_railway
        ;;
    2)
        deploy_digitalocean
        ;;
    3)
        deploy_aws
        ;;
    4)
        test_local
        ;;
    5)
        echo "Goodbye!"
        exit 0
        ;;
    *)
        echo "Invalid option"
        exit 1
        ;;
esac

echo "Deployment complete!"