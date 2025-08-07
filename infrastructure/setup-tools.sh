#!/bin/bash
# Quick setup script for infrastructure tools
set -euo pipefail

echo "Setting up Infrastructure Tools"
echo "=================================="

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    echo "ERROR: Don't run as root! Run as regular user."
    exit 1
fi

# Update package manager
echo "Updating package manager..."
sudo pacman -Sy --noconfirm

# Install basic tools
echo "Installing basic tools..."
sudo pacman -S --noconfirm curl wget unzip jq git

# Install Terraform
echo "Installing Terraform..."
if ! command -v terraform &> /dev/null; then
    cd /tmp
    wget https://releases.hashicorp.com/terraform/1.6.6/terraform_1.6.6_linux_amd64.zip
    unzip terraform_1.6.6_linux_amd64.zip
    sudo mv terraform /usr/local/bin/
    rm terraform_1.6.6_linux_amd64.zip
    echo "SUCCESS: Terraform installed: $(terraform --version)"
else
    echo "SUCCESS: Terraform already installed: $(terraform --version)"
fi

# Install Ansible
echo "Installing Ansible..."
if ! command -v ansible &> /dev/null; then
    pip install --user ansible ansible-lint
    echo "SUCCESS: Ansible installed: $(ansible --version | head -1)"
else
    echo "SUCCESS: Ansible already installed: $(ansible --version | head -1)"
fi

# Install Docker
echo "Installing Docker..."
if ! command -v docker &> /dev/null; then
    sudo pacman -S --noconfirm docker docker-compose
    sudo systemctl enable docker
    sudo systemctl start docker
    sudo usermod -aG docker $USER
    echo "SUCCESS: Docker installed. You'll need to log out and back in for group changes."
else
    echo "SUCCESS: Docker already installed: $(docker --version)"
fi

# Install AWS CLI
echo "Installing AWS CLI..."
if ! command -v aws &> /dev/null; then
    cd /tmp
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
    unzip awscliv2.zip
    sudo ./aws/install
    rm -rf aws awscliv2.zip
    echo "SUCCESS: AWS CLI installed: $(aws --version)"
else
    echo "SUCCESS: AWS CLI already installed: $(aws --version)"
fi

# Install additional tools
echo "Installing additional tools..."
sudo pacman -S --noconfirm yamllint

echo ""
echo "Installation complete!"
echo ""
echo "Next steps:"
echo "1. Log out and back in for Docker group changes"
echo "2. Run: cd /home/ryan/SHOWCASE && ./infrastructure/validate-all.sh"
echo "3. Follow the testing guide in HOW-TO-RUN-EVERYTHING.md"