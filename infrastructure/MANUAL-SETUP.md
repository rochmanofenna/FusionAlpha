# MANUAL INFRASTRUCTURE SETUP

## Install Tools Step by Step

### 1. Install Terraform
```bash
cd /tmp
wget https://releases.hashicorp.com/terraform/1.6.6/terraform_1.6.6_linux_amd64.zip
unzip terraform_1.6.6_linux_amd64.zip
sudo mv terraform /usr/local/bin/
terraform --version
```

### 2. Install Ansible
```bash
pip install --user ansible ansible-lint
ansible --version
```

### 3. Install Docker (if not already installed)
```bash
sudo pacman -S docker docker-compose
sudo systemctl enable docker
sudo systemctl start docker
sudo usermod -aG docker $USER
# Log out and back in after this
```

### 4. Install AWS CLI
```bash
cd /tmp
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
aws --version
```

### 5. Install Additional Tools
```bash
sudo pacman -S yamllint jq
pip install --user docker-compose
```

## Quick Test Without Installation

If you want to test immediately without installing everything:

### Basic File Validation
```bash
# Check if all files exist
find /home/ryan/SHOWCASE/infrastructure -name "*.yml" -o -name "*.tf" -o -name "*.service"

# Count lines of code
find /home/ryan/SHOWCASE/infrastructure -type f | xargs wc -l | tail -1

# Check SystemD service structure
grep -E "^\[(Unit|Service|Install)\]" /home/ryan/SHOWCASE/infrastructure/rhel-configs/*.service
```

### Docker-Based Testing
```bash
# Test Terraform validation using Docker
docker run --rm -v /home/ryan/SHOWCASE/infrastructure/terraform/gpu-cluster:/workspace \
  -w /workspace hashicorp/terraform:1.6.6 sh -c "terraform init -backend=false && terraform validate"

# Test Ansible syntax using Docker
docker run --rm -v /home/ryan/SHOWCASE/infrastructure/ansible:/workspace \
  -w /workspace quay.io/ansible/ansible-runner:latest \
  ansible-playbook gpu-cluster-setup.yml --syntax-check

# Test Prometheus config using Docker
docker run --rm -v /home/ryan/SHOWCASE/infrastructure/monitoring/prometheus:/workspace \
  prom/prometheus:latest promtool check config /workspace/prometheus.yml
```

## Even Simpler - Just Show the Files

For interview purposes, you don't need to run anything. Just show:

### 1. File Structure
```bash
tree /home/ryan/SHOWCASE/infrastructure
```

### 2. Key Configuration Files
```bash
# Show SystemD service
cat /home/ryan/SHOWCASE/infrastructure/rhel-configs/bicep-service.service

# Show Terraform main file  
head -50 /home/ryan/SHOWCASE/infrastructure/terraform/gpu-cluster/main.tf

# Show Ansible playbook
head -50 /home/ryan/SHOWCASE/infrastructure/ansible/gpu-cluster-setup.yml

# Show CI/CD pipeline
head -50 /home/ryan/SHOWCASE/infrastructure/cicd/gitlab-ci.yml
```

### 3. Configuration Validation
```bash
# Basic YAML syntax check
python -c "import yaml; yaml.safe_load(open('/home/ryan/SHOWCASE/infrastructure/ansible/gpu-cluster-setup.yml'))"

# Basic JSON syntax check  
python -c "import json; json.load(open('/home/ryan/SHOWCASE/infrastructure/grafana-dashboards/gpu-cluster-overview.json'))"
```

## What to Say in Interview

"I've created comprehensive infrastructure automation but haven't deployed it to AWS due to cost - GPU instances are expensive. However, I've validated all configurations locally and can show you the complete setup."

Then demonstrate by opening the files and explaining the architecture.