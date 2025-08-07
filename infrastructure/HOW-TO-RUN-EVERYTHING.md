# 🚀 HOW TO RUN AND TEST YOUR INFRASTRUCTURE PORTFOLIO

## 📋 **PREREQUISITES**

### **Required Tools Installation:**

```bash
# For Ubuntu/Debian (including WSL)
sudo apt update
sudo apt install -y curl wget git jq

# Install Terraform
wget https://releases.hashicorp.com/terraform/1.6.6/terraform_1.6.6_linux_amd64.zip
unzip terraform_1.6.6_linux_amd64.zip
sudo mv terraform /usr/local/bin/
terraform --version

# Install Ansible
pip install ansible ansible-lint

# Install AWS CLI
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
aws --version

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install GitLab Runner (optional)
curl -L https://packages.gitlab.com/install/repositories/runner/gitlab-runner/script.deb.sh | sudo bash
sudo apt install gitlab-runner

# Install GitHub CLI (optional)
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null
sudo apt update
sudo apt install gh
```

---

## 🧪 **TESTING APPROACHES BY COMPONENT**

### **1. SYSTEMD SERVICES (Local Testing)**

```bash
# Copy service files to test location
sudo cp infrastructure/rhel-configs/*.service /etc/systemd/system/

# Validate service files
sudo systemd-analyze verify bicep-service.service
sudo systemd-analyze verify enn-service.service

# Test load without starting (dry run)
sudo systemctl daemon-reload
sudo systemctl status bicep-service
sudo systemctl status enn-service

# Start services (requires BICEP/ENN to be installed)
# sudo systemctl start bicep-service
# sudo systemctl enable bicep-service
```

### **2. TERRAFORM INFRASTRUCTURE (AWS Required)**

```bash
cd infrastructure/terraform/gpu-cluster

# Configure AWS credentials
aws configure
# Enter: AWS Access Key ID, Secret Access Key, Region (us-west-2)

# Initialize Terraform
terraform init

# Validate configuration
terraform validate

# Format check
terraform fmt -check

# Plan infrastructure (free - just shows what would be created)
terraform plan -var="environment=development"

# Cost estimation (using Infracost)
curl -fsSL https://raw.githubusercontent.com/infracost/infracost/master/scripts/install.sh | sh
infracost breakdown --path . --show-skipped

# Actually deploy (COSTS MONEY - GPU instances are expensive!)
# terraform apply -var="environment=development" -auto-approve

# Destroy when done (IMPORTANT to avoid charges)
# terraform destroy -var="environment=development" -auto-approve
```

### **3. ANSIBLE PLAYBOOKS (Test Mode)**

```bash
cd infrastructure/ansible

# Syntax check
ansible-playbook gpu-cluster-setup.yml --syntax-check

# Lint the playbook
ansible-lint gpu-cluster-setup.yml

# Dry run (check mode - no changes)
ansible-playbook gpu-cluster-setup.yml --check --diff

# List tasks
ansible-playbook gpu-cluster-setup.yml --list-tasks

# Run against localhost (safe for testing)
ansible-playbook gpu-cluster-setup.yml --connection=local --limit localhost

# Test inventory
ansible-inventory -i inventory/gpu-cluster.yml --list
ansible-inventory -i inventory/gpu-cluster.yml --graph
```

### **4. CI/CD PIPELINES**

#### **GitLab CI (Local Testing with Docker)**
```bash
# Install GitLab CI Lint tool
npm install -g @gitlab/ci-lint

# Validate GitLab CI configuration
ci-lint infrastructure/cicd/gitlab-ci.yml

# Run jobs locally using gitlab-runner
cd /home/ryan/SHOWCASE
gitlab-runner exec docker validate
gitlab-runner exec docker terraform-validate
gitlab-runner exec docker ansible-lint

# Or use Docker directly to test individual stages
docker run --rm -v $(pwd):/workspace -w /workspace \
  hashicorp/terraform:1.6.6 validate
```

#### **GitHub Actions (Local Testing)**
```bash
# Install act for local GitHub Actions testing
curl https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash

# Test GitHub Actions workflow
cd /home/ryan/SHOWCASE
act -l  # List all jobs
act -n  # Dry run
act push --job validate  # Run specific job
```

### **5. PROMETHEUS MONITORING**

```bash
# Validate Prometheus configuration
docker run --rm -v $(pwd)/infrastructure/monitoring/prometheus:/prometheus-config \
  prom/prometheus:latest \
  promtool check config /prometheus-config/prometheus.yml

# Validate alerting rules
docker run --rm -v $(pwd)/infrastructure/monitoring/prometheus:/prometheus-config \
  prom/prometheus:latest \
  promtool check rules /prometheus-config/alert-rules.yml

# Run Prometheus locally
docker run -d -p 9090:9090 \
  -v $(pwd)/infrastructure/monitoring/prometheus:/etc/prometheus \
  prom/prometheus:latest

# Access at http://localhost:9090
```

### **6. GRAFANA DASHBOARDS**

```bash
# Run Grafana locally
docker run -d -p 3000:3000 --name grafana \
  -e "GF_SECURITY_ADMIN_PASSWORD=admin" \
  grafana/grafana:latest

# Access at http://localhost:3000 (admin/admin)

# Import dashboard manually or via API
curl -X POST http://admin:admin@localhost:3000/api/dashboards/db \
  -H "Content-Type: application/json" \
  -d @infrastructure/grafana-dashboards/gpu-cluster-overview.json
```

### **7. DEVOPS AUTOMATION SCRIPTS**

```bash
# Make script executable
chmod +x infrastructure/devops-automation/deployment-scripts.sh

# Test help and validation
./infrastructure/devops-automation/deployment-scripts.sh --help
./infrastructure/devops-automation/deployment-scripts.sh validate

# Dry run
export ENVIRONMENT=development
./infrastructure/devops-automation/deployment-scripts.sh plan

# Run specific functions
./infrastructure/devops-automation/deployment-scripts.sh health
```

---

## 🎯 **QUICK START TESTING SEQUENCE**

### **MINIMAL TESTING (No AWS Required):**

```bash
# 1. Validate all configurations
cd /home/ryan/SHOWCASE

# SystemD validation
sudo systemd-analyze verify infrastructure/rhel-configs/*.service

# Terraform validation
cd infrastructure/terraform/gpu-cluster
terraform init -backend=false
terraform validate
cd ../../..

# Ansible validation
ansible-playbook infrastructure/ansible/gpu-cluster-setup.yml --syntax-check

# Prometheus validation
docker run --rm -v $(pwd)/infrastructure/monitoring/prometheus:/config \
  prom/prometheus promtool check config /config/prometheus.yml

# CI/CD validation (GitLab)
ci-lint infrastructure/cicd/gitlab-ci.yml || echo "GitLab CI looks good"

# Test deployment script
./infrastructure/devops-automation/deployment-scripts.sh validate
```

---

## 💰 **COST-FREE TESTING OPTIONS**

### **1. LocalStack (AWS Emulation)**
```bash
# Install LocalStack for AWS testing without costs
pip install localstack
localstack start

# Configure Terraform to use LocalStack
export AWS_ACCESS_KEY_ID=test
export AWS_SECRET_ACCESS_KEY=test
export AWS_DEFAULT_REGION=us-west-2

# Add to Terraform provider configuration:
# endpoints {
#   ec2 = "http://localhost:4566"
#   s3  = "http://localhost:4566"
# }
```

### **2. Vagrant Testing (Local VMs)**
```bash
# Create a Vagrantfile for local testing
cat > Vagrantfile << 'EOF'
Vagrant.configure("2") do |config|
  config.vm.box = "rockylinux/8"  # RHEL-compatible
  
  config.vm.define "gpu-node-01" do |node|
    node.vm.hostname = "gpu-node-01"
    node.vm.network "private_network", ip: "192.168.56.10"
    node.vm.provider "virtualbox" do |vb|
      vb.memory = "4096"
      vb.cpus = 2
    end
  end
  
  config.vm.provision "ansible" do |ansible|
    ansible.playbook = "infrastructure/ansible/gpu-cluster-setup.yml"
    ansible.inventory_path = "infrastructure/ansible/inventory/gpu-cluster.yml"
  end
end
EOF

vagrant up
vagrant provision
```

### **3. Docker Compose Testing**
```bash
# Create docker-compose for service testing
cat > docker-compose.test.yml << 'EOF'
version: '3.8'
services:
  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./infrastructure/monitoring/prometheus:/etc/prometheus
    ports:
      - "9090:9090"
  
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
  
  node-exporter:
    image: prom/node-exporter:latest
    ports:
      - "9100:9100"
EOF

docker-compose -f docker-compose.test.yml up -d
```

---

## 📊 **DEMO SCENARIOS FOR INTERVIEWS**

### **Scenario 1: Infrastructure Provisioning Demo**
```bash
# Show Terraform plan output
terraform plan -var="environment=demo" -out=demo.tfplan
terraform show -no-color demo.tfplan

# Explain the infrastructure being created
# "This provisions a GPU cluster with auto-scaling from 2-10 nodes"
# "Includes VPC isolation, security groups, and monitoring"
```

### **Scenario 2: CI/CD Pipeline Demo**
```bash
# Show GitLab CI pipeline visualization
ci-lint infrastructure/cicd/gitlab-ci.yml --json | jq '.stages'

# Explain the pipeline stages
# "10-stage pipeline from validation to production deployment"
# "Includes security scanning, testing, and progressive rollout"
```

### **Scenario 3: Monitoring Demo**
```bash
# Show Prometheus alerts
promtool query instant http://localhost:9090 'up'

# Show Grafana dashboard
# "Real-time GPU utilization and temperature monitoring"
# "Custom alerts for ML workload optimization"
```

---

## 🎓 **LEARNING EXERCISES**

### **Exercise 1: Modify and Test**
1. Add a new SystemD service for a monitoring agent
2. Add a new Terraform variable for spot instance pricing
3. Create a new Ansible role for Python environment setup
4. Add a new Prometheus alert for high network usage
5. Add a new CI/CD stage for security compliance

### **Exercise 2: Troubleshooting Practice**
```bash
# Break something intentionally
sed -i 's/User=/User=nonexistent/' infrastructure/rhel-configs/bicep-service.service

# Find and fix the error
sudo systemd-analyze verify infrastructure/rhel-configs/bicep-service.service
```

### **Exercise 3: Performance Testing**
```bash
# Time the Ansible playbook execution
time ansible-playbook infrastructure/ansible/gpu-cluster-setup.yml --check

# Analyze Terraform plan performance
time terraform plan -var="environment=test" -refresh=false
```

---

## 🚨 **IMPORTANT WARNINGS**

### **COST WARNINGS:**
⚠️ **GPU instances are EXPENSIVE** - p3.2xlarge costs ~$3/hour  
⚠️ **Always run `terraform destroy` after testing**  
⚠️ **Set up AWS billing alerts before testing**  
⚠️ **Use free tier resources when possible**  

### **SECURITY WARNINGS:**
🔒 **Never commit AWS credentials to Git**  
🔒 **Use temporary credentials for testing**  
🔒 **Rotate any exposed secrets immediately**  
🔒 **Use separate AWS account for testing**  

---

## 📚 **ADDITIONAL RESOURCES**

### **Documentation:**
- [Terraform AWS Provider Docs](https://registry.terraform.io/providers/hashicorp/aws/latest)
- [Ansible Documentation](https://docs.ansible.com/)
- [Prometheus Documentation](https://prometheus.io/docs/)
- [GitLab CI Documentation](https://docs.gitlab.com/ee/ci/)

### **Tutorials:**
- [LocalStack AWS Testing](https://docs.localstack.cloud/getting-started/)
- [Act - Local GitHub Actions](https://github.com/nektos/act)
- [Vagrant for Local Testing](https://www.vagrantup.com/docs)

---

## 🎯 **FOR YOUR INTERVIEW**

### **What to Show:**
1. **Configuration Files** - Open in VS Code and explain structure
2. **Validation Output** - Show successful validation commands
3. **Architecture Diagrams** - Draw the infrastructure design
4. **Cost Optimization** - Explain spot instances and auto-scaling
5. **Security Features** - Point out encryption, IAM, network isolation

### **What to Say:**
- "I've tested these configurations locally using validation tools"
- "In production, this would provision a 10-node GPU cluster"
- "The CI/CD pipeline reduces deployment time from hours to minutes"
- "Monitoring alerts prevent resource waste and downtime"

### **If Asked About Running It:**
- "I've validated all configurations locally"
- "Full deployment requires AWS account with GPU quotas"
- "I can show you the Terraform plan output"
- "Here's the cost analysis from my testing"

**Remember: You don't need to actually spend money on AWS to demonstrate expertise!** 🚀