#!/bin/bash
# Quick validation script for all infrastructure components
# No AWS account or costs required!

set -euo pipefail

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "🔍 Infrastructure Portfolio Validation Script"
echo "============================================"
echo ""

# Check if running in correct directory
if [ ! -d "infrastructure" ]; then
    echo -e "${RED}❌ Error: Must run from /home/ryan/SHOWCASE directory${NC}"
    exit 1
fi

# Function to check if command exists
check_command() {
    if command -v $1 &> /dev/null; then
        echo -e "${GREEN}✅ $1 is installed${NC}"
        return 0
    else
        echo -e "${YELLOW}⚠️  $1 is not installed (optional for validation)${NC}"
        return 1
    fi
}

# Function to validate files exist
validate_files() {
    local category=$1
    shift
    local files=("$@")
    
    echo ""
    echo "📁 Validating $category files:"
    
    local all_exist=true
    for file in "${files[@]}"; do
        if [ -f "$file" ]; then
            echo -e "  ${GREEN}✅ $file exists${NC}"
        else
            echo -e "  ${RED}❌ $file missing${NC}"
            all_exist=false
        fi
    done
    
    if $all_exist; then
        echo -e "  ${GREEN}All $category files present!${NC}"
    fi
}

# Check dependencies
echo "🔧 Checking optional tools:"
check_command terraform
check_command ansible
check_command docker
check_command aws
check_command jq

# Validate SystemD files
validate_files "SystemD" \
    "infrastructure/rhel-configs/bicep-service.service" \
    "infrastructure/rhel-configs/enn-service.service"

# Basic SystemD validation (works without sudo)
echo ""
echo "🔍 SystemD Syntax Check:"
for service in infrastructure/rhel-configs/*.service; do
    if grep -q "^\[Unit\]" "$service" && grep -q "^\[Service\]" "$service" && grep -q "^\[Install\]" "$service"; then
        echo -e "  ${GREEN}✅ $(basename $service) has valid structure${NC}"
    else
        echo -e "  ${RED}❌ $(basename $service) missing required sections${NC}"
    fi
done

# Validate Terraform files
validate_files "Terraform" \
    "infrastructure/terraform/gpu-cluster/main.tf" \
    "infrastructure/terraform/gpu-cluster/variables.tf" \
    "infrastructure/terraform/gpu-cluster/user_data.sh"

# Basic Terraform validation (if installed)
if command -v terraform &> /dev/null; then
    echo ""
    echo "🔍 Terraform Validation:"
    cd infrastructure/terraform/gpu-cluster
    if terraform init -backend=false &> /dev/null; then
        echo -e "  ${GREEN}✅ Terraform init successful${NC}"
        if terraform validate &> /dev/null; then
            echo -e "  ${GREEN}✅ Terraform configuration is valid${NC}"
        else
            echo -e "  ${RED}❌ Terraform validation failed${NC}"
        fi
    fi
    cd ../../..
fi

# Validate Ansible files
validate_files "Ansible" \
    "infrastructure/ansible/gpu-cluster-setup.yml" \
    "infrastructure/ansible/inventory/gpu-cluster.yml"

# Basic Ansible validation (if installed)
if command -v ansible &> /dev/null; then
    echo ""
    echo "🔍 Ansible Validation:"
    if ansible-playbook infrastructure/ansible/gpu-cluster-setup.yml --syntax-check &> /dev/null; then
        echo -e "  ${GREEN}✅ Ansible playbook syntax is valid${NC}"
    else
        echo -e "  ${RED}❌ Ansible syntax check failed${NC}"
    fi
fi

# Validate Prometheus files
validate_files "Prometheus" \
    "infrastructure/monitoring/prometheus/prometheus.yml" \
    "infrastructure/monitoring/prometheus/alert-rules.yml"

# Validate CI/CD files
validate_files "CI/CD" \
    "infrastructure/cicd/gitlab-ci.yml" \
    "infrastructure/cicd/github-actions.yml"

# Validate DevOps automation
validate_files "DevOps Automation" \
    "infrastructure/devops-automation/deployment-scripts.sh"

# Check if deployment script is executable
if [ -x "infrastructure/devops-automation/deployment-scripts.sh" ]; then
    echo -e "  ${GREEN}✅ Deployment script is executable${NC}"
else
    echo -e "  ${YELLOW}⚠️  Deployment script not executable (run: chmod +x ...)${NC}"
fi

# Validate Grafana dashboards
validate_files "Grafana" \
    "infrastructure/grafana-dashboards/gpu-cluster-overview.json"

# Validate educational content
validate_files "Learning Materials" \
    "infrastructure/learning/systemd-services.txt" \
    "infrastructure/learning/terraform-infrastructure-as-code.txt" \
    "infrastructure/learning/ansible-configuration-management.txt" \
    "infrastructure/learning/devops-cicd-automation.txt"

# Summary
echo ""
echo "📊 Summary Statistics:"
echo "========================"

# Count files
total_files=$(find infrastructure -type f -name "*.yml" -o -name "*.yaml" -o -name "*.tf" -o -name "*.service" -o -name "*.sh" -o -name "*.json" | wc -l)
total_lines=$(find infrastructure -type f -name "*.yml" -o -name "*.yaml" -o -name "*.tf" -o -name "*.service" -o -name "*.sh" -o -name "*.json" | xargs wc -l | tail -1 | awk '{print $1}')

echo -e "Total infrastructure files: ${GREEN}$total_files${NC}"
echo -e "Total lines of code: ${GREEN}$total_lines${NC}"

# List all created files
echo ""
echo "📝 All Infrastructure Files:"
echo "============================"
find infrastructure -type f \( -name "*.yml" -o -name "*.yaml" -o -name "*.tf" -o -name "*.service" -o -name "*.sh" -o -name "*.json" -o -name "*.txt" -o -name "*.md" \) | sort | while read -r file; do
    size=$(wc -l < "$file")
    echo -e "  ${GREEN}✓${NC} $file (${size} lines)"
done

echo ""
echo -e "${GREEN}✅ Validation complete!${NC}"
echo ""
echo "💡 Next Steps:"
echo "  1. Review the HOW-TO-RUN-EVERYTHING.md for detailed testing"
echo "  2. Try running individual validations with installed tools"
echo "  3. Use these files as portfolio evidence in interviews"
echo ""
echo "🎯 For Squarepoint Interview:"
echo "  - Show these files in VS Code or GitHub"
echo "  - Explain the architecture and design decisions"
echo "  - Discuss cost optimization and security features"
echo "  - Mention successful local validation without needing AWS"