# GPU Cluster Infrastructure Variables

variable "aws_region" {
  description = "AWS region for GPU cluster deployment"
  type        = string
  default     = "us-west-2"
  
  validation {
    condition = can(regex("^[a-z]+-[a-z]+-[0-9]+$", var.aws_region))
    error_message = "AWS region must be in format like 'us-west-2'."
  }
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "dev"
  
  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be one of: dev, staging, prod."
  }
}

variable "cluster_name" {
  description = "Name of the GPU cluster"
  type        = string
  default     = "ml-gpu-cluster"
  
  validation {
    condition     = can(regex("^[a-z0-9-]+$", var.cluster_name))
    error_message = "Cluster name must contain only lowercase letters, numbers, and hyphens."
  }
}

variable "cost_center" {
  description = "Cost center for billing allocation"
  type        = string
  default     = "ml-research"
}

variable "owner_email" {
  description = "Email of the cluster owner"
  type        = string
  default     = "ml-team@company.com"
  
  validation {
    condition     = can(regex("^[^@]+@[^@]+\\.[^@]+$", var.owner_email))
    error_message = "Owner email must be a valid email address."
  }
}

# Networking Configuration
variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
  
  validation {
    condition     = can(cidrhost(var.vpc_cidr, 0))
    error_message = "VPC CIDR must be a valid IPv4 CIDR block."
  }
}

variable "availability_zones" {
  description = "List of availability zones to use"
  type        = list(string)
  default     = ["us-west-2a", "us-west-2b", "us-west-2c"]
}

variable "admin_cidr_blocks" {
  description = "CIDR blocks allowed for admin access"
  type        = list(string)
  default     = ["10.0.0.0/8", "172.16.0.0/12"]
}

variable "monitoring_cidr_blocks" {
  description = "CIDR blocks allowed for monitoring access"
  type        = list(string)
  default     = ["10.0.0.0/8"]
}

# Instance Configuration
variable "gpu_instance_type" {
  description = "EC2 instance type for GPU nodes"
  type        = string
  default     = "p3.2xlarge"
  
  validation {
    condition = contains([
      "p3.2xlarge", "p3.8xlarge", "p3.16xlarge", "p3dn.24xlarge",
      "p4d.24xlarge", "g4dn.xlarge", "g4dn.2xlarge", "g4dn.4xlarge",
      "g4dn.8xlarge", "g4dn.12xlarge", "g4dn.16xlarge", "g5.xlarge",
      "g5.2xlarge", "g5.4xlarge", "g5.8xlarge", "g5.12xlarge", "g5.16xlarge"
    ], var.gpu_instance_type)
    error_message = "Instance type must be a valid GPU instance type."
  }
}

variable "key_pair_name" {
  description = "Name of the AWS key pair for SSH access"
  type        = string
  default     = "ml-cluster-key"
}

# Auto Scaling Configuration
variable "min_cluster_size" {
  description = "Minimum number of GPU instances"
  type        = number
  default     = 1
  
  validation {
    condition     = var.min_cluster_size >= 0 && var.min_cluster_size <= 100
    error_message = "Minimum cluster size must be between 0 and 100."
  }
}

variable "max_cluster_size" {
  description = "Maximum number of GPU instances"
  type        = number
  default     = 10
  
  validation {
    condition     = var.max_cluster_size >= 1 && var.max_cluster_size <= 100
    error_message = "Maximum cluster size must be between 1 and 100."
  }
}

variable "desired_cluster_size" {
  description = "Desired number of GPU instances"
  type        = number
  default     = 2
  
  validation {
    condition     = var.desired_cluster_size >= 1 && var.desired_cluster_size <= 100
    error_message = "Desired cluster size must be between 1 and 100."
  }
}

# Storage Configuration
variable "root_volume_size" {
  description = "Size of root EBS volume in GB"
  type        = number
  default     = 100
  
  validation {
    condition     = var.root_volume_size >= 20 && var.root_volume_size <= 16384
    error_message = "Root volume size must be between 20 and 16384 GB."
  }
}

variable "data_volume_size" {
  description = "Size of data EBS volume in GB"
  type        = number
  default     = 1000
  
  validation {
    condition     = var.data_volume_size >= 100 && var.data_volume_size <= 16384
    error_message = "Data volume size must be between 100 and 16384 GB."
  }
}

# S3 Configuration
variable "s3_bucket_artifacts" {
  description = "S3 bucket name for ML artifacts and checkpoints"
  type        = string
  default     = "ml-gpu-cluster-artifacts"
  
  validation {
    condition     = can(regex("^[a-z0-9.-]+$", var.s3_bucket_artifacts))
    error_message = "S3 bucket name must contain only lowercase letters, numbers, dots, and hyphens."
  }
}

# Software Versions
variable "nvidia_driver_version" {
  description = "NVIDIA driver version to install"
  type        = string
  default     = "525.125.06"
}

variable "cuda_version" {
  description = "CUDA toolkit version to install"
  type        = string
  default     = "12.0"
}

variable "docker_version" {
  description = "Docker version to install"
  type        = string
  default     = "24.0.7"
}

# Monitoring Configuration
variable "log_retention_days" {
  description = "CloudWatch log retention in days"
  type        = number
  default     = 30
  
  validation {
    condition = contains([
      1, 3, 5, 7, 14, 30, 60, 90, 120, 150, 180, 365, 400, 545, 731, 1827, 3653
    ], var.log_retention_days)
    error_message = "Log retention days must be a valid CloudWatch retention period."
  }
}

# Cost Optimization
variable "enable_spot_instances" {
  description = "Enable spot instances for cost optimization"
  type        = bool
  default     = false
}

variable "spot_max_price" {
  description = "Maximum spot price per hour"
  type        = string
  default     = "1.50"
}

# Backup Configuration
variable "enable_automated_backups" {
  description = "Enable automated EBS snapshots"
  type        = bool
  default     = true
}

variable "backup_retention_days" {
  description = "Number of days to retain automated backups"
  type        = number
  default     = 7
  
  validation {
    condition     = var.backup_retention_days >= 1 && var.backup_retention_days <= 35
    error_message = "Backup retention days must be between 1 and 35."
  }
}