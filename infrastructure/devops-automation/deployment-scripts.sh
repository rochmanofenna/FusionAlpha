#!/bin/bash
# DevOps Automation Scripts for ML GPU Cluster
# Comprehensive deployment and management automation

set -euo pipefail

# =================================
# CONFIGURATION AND VARIABLES
# =================================

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="/var/log/ml-cluster-deployment"
LOCK_FILE="/tmp/ml-cluster-deployment.lock"

# Environment configuration
ENVIRONMENT="${ENVIRONMENT:-development}"
AWS_REGION="${AWS_REGION:-us-west-2}"
TERRAFORM_DIR="$PROJECT_ROOT/infrastructure/terraform/gpu-cluster"
ANSIBLE_DIR="$PROJECT_ROOT/infrastructure/ansible"

# Application versions
BICEP_VERSION="${BICEP_VERSION:-1.2.3}"
ENN_VERSION="${ENN_VERSION:-2.1.0}"

# Docker registry
REGISTRY="${REGISTRY:-ghcr.io/user/ml-gpu-cluster}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# =================================
# UTILITY FUNCTIONS
# =================================

log() {
    local level="$1"
    shift
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${timestamp} [${level}] $*" | tee -a "$LOG_DIR/deployment.log"
}

log_info() {
    log "INFO" "${BLUE}$*${NC}"
}

log_warn() {
    log "WARN" "${YELLOW}$*${NC}"
}

log_error() {
    log "ERROR" "${RED}$*${NC}"
}

log_success() {
    log "SUCCESS" "${GREEN}$*${NC}"
}

check_dependencies() {
    local dependencies=(
        "terraform:Terraform CLI for infrastructure provisioning"
        "ansible:Ansible for configuration management"
        "docker:Docker for container operations"
        "aws:AWS CLI for cloud operations"
        "kubectl:Kubernetes CLI for container orchestration"
        "helm:Helm for Kubernetes package management"
        "jq:JSON processor for parsing outputs"
        "curl:HTTP client for API calls"
    )
    
    log_info "Checking required dependencies..."
    local missing=()
    
    for dep in "${dependencies[@]}"; do
        local cmd="${dep%%:*}"
        local desc="${dep##*:}"
        
        if ! command -v "$cmd" &> /dev/null; then
            missing+=("$cmd ($desc)")
        fi
    done
    
    if [ ${#missing[@]} -gt 0 ]; then
        log_error "Missing required dependencies:"
        printf '%s\n' "${missing[@]}" | sed 's/^/  - /'
        exit 1
    fi
    
    log_success "All dependencies satisfied"
}

acquire_lock() {
    if [ -f "$LOCK_FILE" ]; then
        local pid=$(cat "$LOCK_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            log_error "Another deployment is already running (PID: $pid)"
            exit 1
        else
            log_warn "Stale lock file found, removing..."
            rm -f "$LOCK_FILE"
        fi
    fi
    
    echo $$ > "$LOCK_FILE"
    trap 'rm -f "$LOCK_FILE"; exit' INT TERM EXIT
}

setup_logging() {
    mkdir -p "$LOG_DIR"
    exec 1> >(tee -a "$LOG_DIR/deployment.log")
    exec 2> >(tee -a "$LOG_DIR/deployment.log" >&2)
}

validate_environment() {
    case "$ENVIRONMENT" in
        development|staging|production)
            log_info "Deploying to environment: $ENVIRONMENT"
            ;;
        *)
            log_error "Invalid environment: $ENVIRONMENT. Must be one of: development, staging, production"
            exit 1
            ;;
    esac
}

# =================================
# INFRASTRUCTURE FUNCTIONS
# =================================

validate_terraform() {
    log_info "Validating Terraform configuration..."
    
    cd "$TERRAFORM_DIR"
    
    # Format check
    if ! terraform fmt -check -recursive; then
        log_error "Terraform format check failed"
        return 1
    fi
    
    # Initialize and validate
    terraform init -backend=false
    terraform validate
    
    log_success "Terraform validation passed"
}

plan_infrastructure() {
    log_info "Planning infrastructure changes for $ENVIRONMENT..."
    
    cd "$TERRAFORM_DIR"
    
    # Initialize with backend
    terraform init
    
    # Create plan
    terraform plan \
        -var="environment=$ENVIRONMENT" \
        -var="bicep_version=$BICEP_VERSION" \
        -var="enn_version=$ENN_VERSION" \
        -out="$ENVIRONMENT.tfplan"
    
    # Save plan summary
    terraform show -no-color "$ENVIRONMENT.tfplan" > "$ENVIRONMENT-plan.txt"
    
    log_success "Infrastructure plan created: $ENVIRONMENT.tfplan"
}

apply_infrastructure() {
    log_info "Applying infrastructure changes for $ENVIRONMENT..."
    
    cd "$TERRAFORM_DIR"
    
    # Apply the plan
    terraform apply -auto-approve "$ENVIRONMENT.tfplan"
    
    # Save outputs
    terraform output -json > "$PROJECT_ROOT/terraform-outputs.json"
    
    log_success "Infrastructure deployment completed"
}

destroy_infrastructure() {
    log_warn "Destroying infrastructure for $ENVIRONMENT..."
    
    read -p "Are you sure you want to destroy the $ENVIRONMENT infrastructure? [y/N]: " -n 1 -r
    echo
    
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Infrastructure destruction cancelled"
        return 0
    fi
    
    cd "$TERRAFORM_DIR"
    
    terraform destroy \
        -var="environment=$ENVIRONMENT" \
        -auto-approve
    
    log_success "Infrastructure destroyed"
}

# =================================
# CONFIGURATION MANAGEMENT FUNCTIONS
# =================================

validate_ansible() {
    log_info "Validating Ansible configuration..."
    
    cd "$ANSIBLE_DIR"
    
    # Lint playbooks
    ansible-lint gpu-cluster-setup.yml
    
    # Validate inventory
    ansible-inventory -i inventory/gpu-cluster.yml --list > /dev/null
    
    log_success "Ansible validation passed"
}

configure_infrastructure() {
    log_info "Configuring infrastructure with Ansible..."
    
    cd "$ANSIBLE_DIR"
    
    # Determine target hosts based on environment
    local limit_hosts=""
    case "$ENVIRONMENT" in
        development)
            limit_hosts="--limit development"
            ;;
        staging)
            limit_hosts="--limit staging"
            ;;
        production)
            limit_hosts="--limit gpu_cluster"
            ;;
    esac
    
    # Run the playbook
    ansible-playbook \
        -i inventory/gpu-cluster.yml \
        gpu-cluster-setup.yml \
        --extra-vars "environment=$ENVIRONMENT" \
        --extra-vars "@$PROJECT_ROOT/terraform-outputs.json" \
        $limit_hosts \
        --diff
    
    log_success "Infrastructure configuration completed"
}

# =================================
# APPLICATION DEPLOYMENT FUNCTIONS
# =================================

build_containers() {
    log_info "Building application containers..."
    
    # Build BICEP container
    log_info "Building BICEP container..."
    cd "$PROJECT_ROOT/BICEP"
    docker build \
        -t "$REGISTRY/bicep:$BICEP_VERSION" \
        -t "$REGISTRY/bicep:latest" \
        .
    
    # Build ENN container
    log_info "Building ENN container..."
    cd "$PROJECT_ROOT/ENN"
    docker build \
        -t "$REGISTRY/enn:$ENN_VERSION" \
        -t "$REGISTRY/enn:latest" \
        .
    
    log_success "Container builds completed"
}

push_containers() {
    log_info "Pushing containers to registry..."
    
    # Push BICEP containers
    docker push "$REGISTRY/bicep:$BICEP_VERSION"
    docker push "$REGISTRY/bicep:latest"
    
    # Push ENN containers
    docker push "$REGISTRY/enn:$ENN_VERSION"
    docker push "$REGISTRY/enn:latest"
    
    log_success "Container push completed"
}

deploy_applications() {
    log_info "Deploying applications to $ENVIRONMENT..."
    
    cd "$ANSIBLE_DIR"
    
    # Deploy BICEP service
    ansible-playbook \
        -i inventory/gpu-cluster.yml \
        deploy-bicep.yml \
        --extra-vars "environment=$ENVIRONMENT" \
        --extra-vars "bicep_version=$BICEP_VERSION" \
        --extra-vars "registry_url=$REGISTRY"
    
    # Deploy ENN service
    ansible-playbook \
        -i inventory/gpu-cluster.yml \
        deploy-enn.yml \
        --extra-vars "environment=$ENVIRONMENT" \
        --extra-vars "enn_version=$ENN_VERSION" \
        --extra-vars "registry_url=$REGISTRY"
    
    log_success "Application deployment completed"
}

# =================================
# TESTING AND VALIDATION FUNCTIONS
# =================================

run_unit_tests() {
    log_info "Running unit tests..."
    
    local test_results=()
    
    # Test BICEP
    cd "$PROJECT_ROOT/BICEP"
    if python -m pytest tests/ -v --cov=src/ --cov-report=xml; then
        test_results+=("BICEP: PASSED")
    else
        test_results+=("BICEP: FAILED")
    fi
    
    # Test ENN
    cd "$PROJECT_ROOT/ENN"
    if python -m pytest tests/ -v --cov=enn/ --cov-report=xml; then
        test_results+=("ENN: PASSED")
    else
        test_results+=("ENN: FAILED")
    fi
    
    # Test Gesture Recognition
    cd "$PROJECT_ROOT/GestureRecognition"
    if python -m pytest tests/ -v --cov=src/ --cov-report=xml; then
        test_results+=("GestureRecognition: PASSED")
    else
        test_results+=("GestureRecognition: FAILED")
    fi
    
    # Report results
    log_info "Unit test results:"
    for result in "${test_results[@]}"; do
        if [[ $result == *"PASSED"* ]]; then
            log_success "  $result"
        else
            log_error "  $result"
        fi
    done
}

run_integration_tests() {
    log_info "Running integration tests against $ENVIRONMENT..."
    
    cd "$PROJECT_ROOT"
    
    # Health check tests
    python tests/integration/test_health_checks.py --environment "$ENVIRONMENT"
    
    # API integration tests
    python tests/integration/test_api_endpoints.py --environment "$ENVIRONMENT"
    
    # GPU cluster tests
    python tests/integration/test_gpu_cluster.py --environment "$ENVIRONMENT"
    
    log_success "Integration tests completed"
}

run_performance_tests() {
    log_info "Running performance tests..."
    
    # GPU benchmark
    python "$PROJECT_ROOT/tests/performance/gpu_benchmark.py" \
        --environment "$ENVIRONMENT" \
        --output "$LOG_DIR/gpu-benchmark-$(date +%Y%m%d-%H%M%S).json"
    
    # Load testing
    k6 run "$PROJECT_ROOT/tests/performance/load-test.js" \
        --env ENVIRONMENT="$ENVIRONMENT" \
        --out json="$LOG_DIR/load-test-$(date +%Y%m%d-%H%M%S).json"
    
    log_success "Performance tests completed"
}

# =================================
# MONITORING AND MAINTENANCE FUNCTIONS
# =================================

deploy_monitoring() {
    log_info "Deploying monitoring stack..."
    
    cd "$ANSIBLE_DIR"
    
    # Deploy Prometheus
    ansible-playbook \
        -i inventory/gpu-cluster.yml \
        deploy-prometheus.yml \
        --extra-vars "environment=$ENVIRONMENT"
    
    # Deploy Grafana
    ansible-playbook \
        -i inventory/gpu-cluster.yml \
        deploy-grafana.yml \
        --extra-vars "environment=$ENVIRONMENT"
    
    log_success "Monitoring stack deployed"
}

backup_data() {
    log_info "Creating backup for $ENVIRONMENT..."
    
    local backup_timestamp=$(date +%Y%m%d-%H%M%S)
    local backup_name="ml-cluster-backup-$ENVIRONMENT-$backup_timestamp"
    
    # Backup Terraform state
    aws s3 cp \
        "$TERRAFORM_DIR/terraform.tfstate" \
        "s3://ml-infrastructure-backups/terraform/$backup_name.tfstate"
    
    # Backup application data
    ansible-playbook \
        -i "$ANSIBLE_DIR/inventory/gpu-cluster.yml" \
        "$ANSIBLE_DIR/backup-data.yml" \
        --extra-vars "backup_name=$backup_name" \
        --extra-vars "environment=$ENVIRONMENT"
    
    log_success "Backup completed: $backup_name"
}

health_check() {
    log_info "Performing health check for $ENVIRONMENT..."
    
    local health_status=()
    
    # Check infrastructure
    if terraform show -json "$TERRAFORM_DIR/terraform.tfstate" | jq -e '.values' > /dev/null 2>&1; then
        health_status+=("Infrastructure: HEALTHY")
    else
        health_status+=("Infrastructure: UNHEALTHY")
    fi
    
    # Check services
    cd "$ANSIBLE_DIR"
    if ansible all -i inventory/gpu-cluster.yml -m ping > /dev/null 2>&1; then
        health_status+=("Services: HEALTHY")
    else
        health_status+=("Services: UNHEALTHY")
    fi
    
    # Report health status
    log_info "Health check results:"
    for status in "${health_status[@]}"; do
        if [[ $status == *"HEALTHY"* ]]; then
            log_success "  $status"
        else
            log_error "  $status"
        fi
    done
}

# =================================
# MAIN DEPLOYMENT WORKFLOWS
# =================================

full_deployment() {
    log_info "Starting full deployment workflow for $ENVIRONMENT..."
    
    # Pre-deployment validation
    check_dependencies
    validate_environment
    validate_terraform
    validate_ansible
    
    # Infrastructure deployment
    plan_infrastructure
    apply_infrastructure
    
    # Configuration management
    configure_infrastructure
    
    # Application deployment
    build_containers
    push_containers
    deploy_applications
    
    # Monitoring setup
    deploy_monitoring
    
    # Post-deployment validation
    run_integration_tests
    health_check
    
    log_success "Full deployment completed successfully!"
}

rolling_update() {
    log_info "Starting rolling update for $ENVIRONMENT..."
    
    # Build and push new containers
    build_containers
    push_containers
    
    # Rolling update deployment
    deploy_applications
    
    # Validate deployment
    run_integration_tests
    health_check
    
    log_success "Rolling update completed successfully!"
}

disaster_recovery() {
    log_info "Starting disaster recovery for $ENVIRONMENT..."
    
    # Restore from backup
    local latest_backup=$(aws s3 ls s3://ml-infrastructure-backups/terraform/ | sort | tail -n 1 | awk '{print $4}')
    
    if [ -n "$latest_backup" ]; then
        log_info "Restoring from backup: $latest_backup"
        
        # Download backup
        aws s3 cp "s3://ml-infrastructure-backups/terraform/$latest_backup" "$TERRAFORM_DIR/terraform.tfstate"
        
        # Apply infrastructure
        apply_infrastructure
        configure_infrastructure
        deploy_applications
        
        log_success "Disaster recovery completed"
    else
        log_error "No backup found for disaster recovery"
        exit 1
    fi
}

# =================================
# COMMAND LINE INTERFACE
# =================================

show_usage() {
    cat << EOF
ML GPU Cluster Deployment Automation

Usage: $0 <command> [options]

Commands:
  validate           Validate all configurations
  plan              Plan infrastructure changes
  deploy            Full deployment workflow
  update            Rolling update of applications
  destroy           Destroy infrastructure
  test              Run all tests
  backup            Create backup
  health            Health check
  monitor           Deploy monitoring stack
  recover           Disaster recovery

Options:
  -e, --environment ENV    Target environment (development|staging|production)
  -v, --version VERSION    Application version to deploy
  -h, --help              Show this help message

Examples:
  $0 deploy --environment production
  $0 update --environment staging --version 2.1.1
  $0 health --environment development

EOF
}

main() {
    # Setup
    setup_logging
    acquire_lock
    
    # Parse command line arguments
    local command=""
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -e|--environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -v|--version)
                BICEP_VERSION="$2"
                ENN_VERSION="$2"
                shift 2
                ;;
            -h|--help)
                show_usage
                exit 0
                ;;
            validate|plan|deploy|update|destroy|test|backup|health|monitor|recover)
                command="$1"
                shift
                ;;
            *)
                log_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    if [ -z "$command" ]; then
        log_error "No command specified"
        show_usage
        exit 1
    fi
    
    # Execute command
    case "$command" in
        validate)
            validate_terraform
            validate_ansible
            ;;
        plan)
            plan_infrastructure
            ;;
        deploy)
            full_deployment
            ;;
        update)
            rolling_update
            ;;
        destroy)
            destroy_infrastructure
            ;;
        test)
            run_unit_tests
            run_integration_tests
            run_performance_tests
            ;;
        backup)
            backup_data
            ;;
        health)
            health_check
            ;;
        monitor)
            deploy_monitoring
            ;;
        recover)
            disaster_recovery
            ;;
        *)
            log_error "Unknown command: $command"
            exit 1
            ;;
    esac
}

# Execute main function if script is run directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi