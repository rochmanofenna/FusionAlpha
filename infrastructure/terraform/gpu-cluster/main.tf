# GPU Cluster Infrastructure for ML/HPC Workloads
# Provisions AWS EC2 instances with NVIDIA GPUs for BICEP/ENN training

terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
  
  backend "s3" {
    bucket         = "ml-infrastructure-terraform-state"
    key            = "gpu-cluster/terraform.tfstate"
    region         = "us-west-2"
    encrypt        = true
    dynamodb_table = "terraform-state-lock"
  }
}

provider "aws" {
  region = var.aws_region
  
  default_tags {
    tags = {
      Environment     = var.environment
      Project         = "ML-GPU-Cluster"
      ManagedBy      = "Terraform"
      CostCenter     = var.cost_center
      Owner          = var.owner_email
      BackupSchedule = "daily"
    }
  }
}

# Data sources for latest AMIs and availability zones
data "aws_ami" "ubuntu_gpu" {
  most_recent = true
  owners      = ["099720109477"] # Canonical
  
  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-22.04-amd64-server-*"]
  }
  
  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

data "aws_availability_zones" "available" {
  state = "available"
  
  filter {
    name   = "opt-in-status"
    values = ["opt-in-not-required"]
  }
}

# VPC and Networking
resource "aws_vpc" "gpu_cluster_vpc" {
  cidr_block           = var.vpc_cidr
  enable_dns_hostnames = true
  enable_dns_support   = true
  
  tags = {
    Name = "${var.cluster_name}-vpc"
  }
}

resource "aws_subnet" "gpu_cluster_subnet" {
  count = length(var.availability_zones)
  
  vpc_id                  = aws_vpc.gpu_cluster_vpc.id
  cidr_block              = cidrsubnet(var.vpc_cidr, 8, count.index)
  availability_zone       = data.aws_availability_zones.available.names[count.index]
  map_public_ip_on_launch = true
  
  tags = {
    Name = "${var.cluster_name}-subnet-${count.index + 1}"
    Type = "public"
  }
}

resource "aws_internet_gateway" "gpu_cluster_igw" {
  vpc_id = aws_vpc.gpu_cluster_vpc.id
  
  tags = {
    Name = "${var.cluster_name}-igw"
  }
}

resource "aws_route_table" "gpu_cluster_rt" {
  vpc_id = aws_vpc.gpu_cluster_vpc.id
  
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.gpu_cluster_igw.id
  }
  
  tags = {
    Name = "${var.cluster_name}-rt"
  }
}

resource "aws_route_table_association" "gpu_cluster_rta" {
  count = length(aws_subnet.gpu_cluster_subnet)
  
  subnet_id      = aws_subnet.gpu_cluster_subnet[count.index].id
  route_table_id = aws_route_table.gpu_cluster_rt.id
}

# Security Groups
resource "aws_security_group" "gpu_cluster_sg" {
  name_prefix = "${var.cluster_name}-gpu-cluster-"
  vpc_id      = aws_vpc.gpu_cluster_vpc.id
  description = "Security group for GPU cluster nodes"
  
  # SSH access from bastion/admin networks
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = var.admin_cidr_blocks
    description = "SSH access for administration"
  }
  
  # NVIDIA NCCL communication for multi-GPU training
  ingress {
    from_port = 0
    to_port   = 65535
    protocol  = "tcp"
    self      = true
    description = "Internal cluster communication"
  }
  
  # NVIDIA NCCL UDP communication
  ingress {
    from_port = 0
    to_port   = 65535
    protocol  = "udp" 
    self      = true
    description = "Internal cluster UDP communication"
  }
  
  # TensorBoard and Jupyter access
  ingress {
    from_port   = 6006
    to_port     = 6010
    protocol    = "tcp"
    cidr_blocks = var.admin_cidr_blocks
    description = "TensorBoard access"
  }
  
  ingress {
    from_port   = 8888
    to_port     = 8892
    protocol    = "tcp"
    cidr_blocks = var.admin_cidr_blocks
    description = "Jupyter notebook access"
  }
  
  # Ray cluster communication (for distributed computing)
  ingress {
    from_port   = 10001
    to_port     = 10003
    protocol    = "tcp"
    self        = true
    description = "Ray cluster communication"
  }
  
  # Prometheus monitoring
  ingress {
    from_port   = 9090
    to_port     = 9100
    protocol    = "tcp"
    cidr_blocks = var.monitoring_cidr_blocks
    description = "Prometheus monitoring"
  }
  
  # All outbound traffic
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
    description = "All outbound traffic"
  }
  
  tags = {
    Name = "${var.cluster_name}-gpu-cluster-sg"
  }
}

# Launch Template for GPU instances
resource "aws_launch_template" "gpu_cluster_template" {
  name_prefix   = "${var.cluster_name}-gpu-"
  image_id      = data.aws_ami.ubuntu_gpu.id
  instance_type = var.gpu_instance_type
  key_name      = var.key_pair_name
  
  vpc_security_group_ids = [aws_security_group.gpu_cluster_sg.id]
  
  # Enhanced monitoring
  monitoring {
    enabled = true
  }
  
  # EBS optimization for high I/O
  ebs_optimized = true
  
  # Instance metadata service v2
  metadata_options {
    http_endpoint = "enabled"
    http_tokens   = "required"
    http_put_response_hop_limit = 1
    instance_metadata_tags      = "enabled"
  }
  
  # GPU-optimized EBS configuration
  block_device_mappings {
    device_name = "/dev/sda1"
    ebs {
      volume_type           = "gp3"
      volume_size           = var.root_volume_size
      iops                  = 3000
      throughput            = 125
      encrypted             = true
      delete_on_termination = true
    }
  }
  
  # Data volume for datasets and checkpoints
  block_device_mappings {
    device_name = "/dev/sdf"
    ebs {
      volume_type           = "gp3"
      volume_size           = var.data_volume_size
      iops                  = 16000
      throughput            = 1000
      encrypted             = true
      delete_on_termination = false
    }
  }
  
  # User data script for GPU setup
  user_data = base64encode(templatefile("${path.module}/user_data.sh", {
    cluster_name    = var.cluster_name
    environment     = var.environment
    s3_bucket       = var.s3_bucket_artifacts
    nvidia_driver   = var.nvidia_driver_version
    cuda_version    = var.cuda_version
    docker_version  = var.docker_version
  }))
  
  tags = {
    Name = "${var.cluster_name}-gpu-template"
  }
  
  tag_specifications {
    resource_type = "instance"
    tags = {
      Name = "${var.cluster_name}-gpu-node"
      Type = "gpu-compute"
    }
  }
  
  tag_specifications {
    resource_type = "volume"
    tags = {
      Name = "${var.cluster_name}-gpu-volume"
      Type = "gpu-storage"
    }
  }
}

# Auto Scaling Group
resource "aws_autoscaling_group" "gpu_cluster_asg" {
  name                = "${var.cluster_name}-gpu-asg"
  vpc_zone_identifier = aws_subnet.gpu_cluster_subnet[*].id
  target_group_arns   = []
  health_check_type   = "EC2"
  health_check_grace_period = 300
  
  min_size         = var.min_cluster_size
  max_size         = var.max_cluster_size
  desired_capacity = var.desired_cluster_size
  
  # Instance refresh for rolling updates
  instance_refresh {
    strategy = "Rolling"
    preferences {
      min_healthy_percentage = 50
      instance_warmup        = 300
    }
  }
  
  launch_template {
    id      = aws_launch_template.gpu_cluster_template.id
    version = "$Latest"
  }
  
  # Scaling policies based on GPU utilization
  enabled_metrics = [
    "GroupMinSize",
    "GroupMaxSize", 
    "GroupDesiredCapacity",
    "GroupInServiceInstances",
    "GroupTotalInstances"
  ]
  
  tag {
    key                 = "Name"
    value               = "${var.cluster_name}-gpu-asg"
    propagate_at_launch = false
  }
  
  tag {
    key                 = "Environment"
    value               = var.environment
    propagate_at_launch = true
  }
  
  tag {
    key                 = "Project"
    value               = "ML-GPU-Cluster"
    propagate_at_launch = true
  }
}

# Application Load Balancer for distributed services
resource "aws_lb" "gpu_cluster_alb" {
  name               = "${var.cluster_name}-gpu-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.gpu_cluster_sg.id]
  subnets            = aws_subnet.gpu_cluster_subnet[*].id
  
  enable_deletion_protection = false
  
  tags = {
    Name = "${var.cluster_name}-gpu-alb"
  }
}

# S3 bucket for artifacts and checkpoints
resource "aws_s3_bucket" "ml_artifacts" {
  bucket = var.s3_bucket_artifacts
  
  tags = {
    Name = "${var.cluster_name}-ml-artifacts"
  }
}

resource "aws_s3_bucket_versioning" "ml_artifacts_versioning" {
  bucket = aws_s3_bucket.ml_artifacts.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_encryption" "ml_artifacts_encryption" {
  bucket = aws_s3_bucket.ml_artifacts.id
  
  server_side_encryption_configuration {
    rule {
      apply_server_side_encryption_by_default {
        sse_algorithm = "AES256"
      }
    }
  }
}

# CloudWatch Log Group for cluster logging
resource "aws_cloudwatch_log_group" "gpu_cluster_logs" {
  name              = "/aws/ec2/${var.cluster_name}"
  retention_in_days = var.log_retention_days
  
  tags = {
    Name = "${var.cluster_name}-logs"
  }
}