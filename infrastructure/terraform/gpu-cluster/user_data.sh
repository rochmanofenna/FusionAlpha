#!/bin/bash
# GPU Cluster Node Initialization Script
# Configures Ubuntu instances with NVIDIA drivers, CUDA, Docker, and ML frameworks

set -euo pipefail

# Configuration from Terraform
CLUSTER_NAME="${cluster_name}"
ENVIRONMENT="${environment}"
S3_BUCKET="${s3_bucket}"
NVIDIA_DRIVER="${nvidia_driver}"
CUDA_VERSION="${cuda_version}"
DOCKER_VERSION="${docker_version}"

# Logging setup
exec 1> >(logger -s -t $(basename $0)) 2>&1
echo "Starting GPU cluster node initialization"
echo "Cluster: $CLUSTER_NAME, Environment: $ENVIRONMENT"

# Update system packages
echo "Updating system packages..."
apt-get update -y
apt-get upgrade -y

# Install essential packages
echo "Installing essential packages..."
apt-get install -y \
    curl \
    wget \
    gnupg \
    software-properties-common \
    apt-transport-https \
    ca-certificates \
    unzip \
    htop \
    iotop \
    nvtop \
    tree \
    git \
    build-essential \
    python3-pip \
    python3-venv \
    awscli \
    jq

# Install CloudWatch agent
echo "Installing CloudWatch agent..."
wget https://s3.amazonaws.com/amazoncloudwatch-agent/ubuntu/amd64/latest/amazon-cloudwatch-agent.deb
dpkg -i amazon-cloudwatch-agent.deb
rm amazon-cloudwatch-agent.deb

# Configure CloudWatch agent
cat > /opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json << 'EOF'
{
  "agent": {
    "metrics_collection_interval": 60,
    "run_as_user": "cwagent"
  },
  "metrics": {
    "namespace": "GPU-Cluster/EC2",
    "metrics_collected": {
      "cpu": {
        "measurement": [
          "cpu_usage_idle",
          "cpu_usage_iowait",
          "cpu_usage_user",
          "cpu_usage_system"
        ],
        "metrics_collection_interval": 60
      },
      "disk": {
        "measurement": [
          "used_percent"
        ],
        "metrics_collection_interval": 60,
        "resources": [
          "*"
        ]
      },
      "diskio": {
        "measurement": [
          "io_time",
          "read_bytes",
          "write_bytes",
          "reads",
          "writes"
        ],
        "metrics_collection_interval": 60,
        "resources": [
          "*"
        ]
      },
      "mem": {
        "measurement": [
          "mem_used_percent"
        ],
        "metrics_collection_interval": 60
      },
      "netstat": {
        "measurement": [
          "tcp_established",
          "tcp_time_wait"
        ],
        "metrics_collection_interval": 60
      },
      "swap": {
        "measurement": [
          "swap_used_percent"
        ],
        "metrics_collection_interval": 60
      }
    }
  },
  "logs": {
    "logs_collected": {
      "files": {
        "collect_list": [
          {
            "file_path": "/var/log/cloud-init.log",
            "log_group_name": "/aws/ec2/${CLUSTER_NAME}",
            "log_stream_name": "{instance_id}/cloud-init.log"
          },
          {
            "file_path": "/var/log/syslog",
            "log_group_name": "/aws/ec2/${CLUSTER_NAME}",
            "log_stream_name": "{instance_id}/syslog"
          },
          {
            "file_path": "/var/log/nvidia-installer.log",
            "log_group_name": "/aws/ec2/${CLUSTER_NAME}",
            "log_stream_name": "{instance_id}/nvidia-installer.log"
          }
        ]
      }
    }
  }
}
EOF

# Start CloudWatch agent
systemctl enable amazon-cloudwatch-agent
systemctl start amazon-cloudwatch-agent

# Format and mount data volume
echo "Configuring data volume..."
if [ -b /dev/nvme1n1 ]; then  # NVMe SSD
    DATA_DEVICE="/dev/nvme1n1"
elif [ -b /dev/xvdf ]; then  # Standard EBS
    DATA_DEVICE="/dev/xvdf"
else
    echo "Warning: No data volume found"
    DATA_DEVICE=""
fi

if [ -n "$DATA_DEVICE" ]; then
    # Format if not already formatted
    if ! blkid $DATA_DEVICE; then
        echo "Formatting data volume $DATA_DEVICE..."
        mkfs.ext4 -F $DATA_DEVICE
    fi
    
    # Create mount point and mount
    mkdir -p /data
    mount $DATA_DEVICE /data
    
    # Add to fstab for persistent mounting
    echo "$DATA_DEVICE /data ext4 defaults,nofail 0 2" >> /etc/fstab
    
    # Set permissions
    chmod 755 /data
    
    # Create standard directories
    mkdir -p /data/{datasets,checkpoints,logs,experiments,models}
    chmod 755 /data/{datasets,checkpoints,logs,experiments,models}
fi

# Install NVIDIA drivers
echo "Installing NVIDIA drivers version $NVIDIA_DRIVER..."
# Add NVIDIA package repository
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
dpkg -i cuda-keyring_1.0-1_all.deb
rm cuda-keyring_1.0-1_all.deb

# Update package list
apt-get update -y

# Install specific NVIDIA driver version
apt-get install -y nvidia-driver-$NVIDIA_DRIVER nvidia-utils-$NVIDIA_DRIVER

# Install CUDA toolkit
echo "Installing CUDA toolkit version $CUDA_VERSION..."
apt-get install -y cuda-toolkit-$(echo $CUDA_VERSION | tr '.' '-')

# Install cuDNN
echo "Installing cuDNN..."
apt-get install -y libcudnn8 libcudnn8-dev

# Install Docker
echo "Installing Docker version $DOCKER_VERSION..."
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | apt-key add -
add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
apt-get update -y
apt-get install -y docker-ce=5:$DOCKER_VERSION~3-0~ubuntu-jammy docker-ce-cli=5:$DOCKER_VERSION~3-0~ubuntu-jammy containerd.io

# Install NVIDIA Container Toolkit
echo "Installing NVIDIA Container Toolkit..."
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | tee /etc/apt/sources.list.d/nvidia-docker.list
apt-get update -y
apt-get install -y nvidia-container-toolkit

# Configure Docker daemon
cat > /etc/docker/daemon.json << 'EOF'
{
    "default-runtime": "nvidia",
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    },
    "log-driver": "json-file",
    "log-opts": {
        "max-size": "10m",
        "max-file": "3"
    },
    "storage-driver": "overlay2"
}
EOF

# Enable and start Docker
systemctl enable docker
systemctl start docker

# Add ubuntu user to docker group
usermod -aG docker ubuntu

# Install Python ML frameworks
echo "Installing Python ML frameworks..."
pip3 install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other ML frameworks
pip3 install \
    tensorflow-gpu \
    transformers \
    datasets \
    accelerate \
    wandb \
    tensorboard \
    jupyter \
    jupyterlab \
    matplotlib \
    seaborn \
    pandas \
    numpy \
    scikit-learn \
    opencv-python \
    pillow \
    tqdm \
    psutil \
    gpustat

# Install monitoring tools
pip3 install \
    prometheus-client \
    nvidia-ml-py3 \
    py3nvml

# Create ML user and directories
echo "Creating ML user and directories..."
useradd -m -s /bin/bash mluser
usermod -aG docker mluser

# Create project directories
mkdir -p /opt/{bicep,enn,projects}
chown -R mluser:mluser /opt/{bicep,enn,projects}
chmod 755 /opt/{bicep,enn,projects}

# Set up environment variables
cat >> /etc/environment << EOF
CUDA_HOME=/usr/local/cuda
PATH="/usr/local/cuda/bin:\$PATH"
LD_LIBRARY_PATH="/usr/local/cuda/lib64:\$LD_LIBRARY_PATH"
NVIDIA_VISIBLE_DEVICES=all
NVIDIA_DRIVER_CAPABILITIES=compute,utility
EOF

# Create systemd service for GPU monitoring
cat > /etc/systemd/system/gpu-monitor.service << 'EOF'
[Unit]
Description=GPU Monitoring Service
After=multi-user.target

[Service]
Type=simple
User=mluser
ExecStart=/usr/local/bin/python3 /opt/scripts/gpu_monitor.py
Restart=always
RestartSec=30

[Install]
WantedBy=multi-user.target
EOF

# Create GPU monitoring script
mkdir -p /opt/scripts
cat > /opt/scripts/gpu_monitor.py << 'EOF'
#!/usr/bin/env python3
import time
import json
import subprocess
import psutil
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_gpu_metrics():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.used,memory.free', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        
        metrics = []
        for line in result.stdout.strip().split('\n'):
            if line:
                values = line.split(', ')
                metrics.append({
                    'gpu_id': int(values[0]),
                    'name': values[1],
                    'temperature': float(values[2]),
                    'utilization_gpu': float(values[3]),
                    'utilization_memory': float(values[4]),
                    'memory_total': float(values[5]),
                    'memory_used': float(values[6]),
                    'memory_free': float(values[7]),
                    'timestamp': datetime.now().isoformat()
                })
        
        return metrics
    except Exception as e:
        logger.error(f"Failed to get GPU metrics: {e}")
        return []

def main():
    while True:
        try:
            gpu_metrics = get_gpu_metrics()
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            system_metrics = {
                'timestamp': datetime.now().isoformat(),
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available_gb': memory.available / (1024**3),
                'gpu_metrics': gpu_metrics
            }
            
            # Log metrics (can be picked up by CloudWatch agent)
            logger.info(f"METRICS: {json.dumps(system_metrics)}")
            
            time.sleep(60)  # Report every minute
            
        except Exception as e:
            logger.error(f"Error in monitoring loop: {e}")
            time.sleep(30)

if __name__ == "__main__":
    main()
EOF

chmod +x /opt/scripts/gpu_monitor.py
chown mluser:mluser /opt/scripts/gpu_monitor.py

# Enable GPU monitoring service
systemctl enable gpu-monitor
systemctl start gpu-monitor

# Download and install project repositories
echo "Setting up project repositories..."
cd /opt

# Clone/setup BICEP project
if [ -n "$S3_BUCKET" ]; then
    aws s3 cp s3://$S3_BUCKET/projects/bicep.tar.gz /tmp/ || echo "BICEP archive not found in S3"
    if [ -f /tmp/bicep.tar.gz ]; then
        tar -xzf /tmp/bicep.tar.gz -C /opt/
        chown -R mluser:mluser /opt/bicep
    fi
fi

# Clone/setup ENN project  
if [ -n "$S3_BUCKET" ]; then
    aws s3 cp s3://$S3_BUCKET/projects/enn.tar.gz /tmp/ || echo "ENN archive not found in S3"
    if [ -f /tmp/enn.tar.gz ]; then
        tar -xzf /tmp/enn.tar.gz -C /opt/
        chown -R mluser:mluser /opt/enn
    fi
fi

# Create Jupyter configuration
mkdir -p /home/mluser/.jupyter
cat > /home/mluser/.jupyter/jupyter_lab_config.py << 'EOF'
c.ServerApp.ip = '0.0.0.0'
c.ServerApp.port = 8888
c.ServerApp.allow_root = False
c.ServerApp.open_browser = False
c.ServerApp.token = ''
c.ServerApp.password = ''
EOF
chown -R mluser:mluser /home/mluser/.jupyter

# Create startup script for ML services
cat > /opt/scripts/start_ml_services.sh << 'EOF'
#!/bin/bash
# Start ML development services

# Start Jupyter Lab
su - mluser -c "cd /opt && nohup jupyter lab --config=/home/mluser/.jupyter/jupyter_lab_config.py > /var/log/jupyter.log 2>&1 &"

# Start TensorBoard (if tensorboard logs exist)
if [ -d /data/logs ]; then
    su - mluser -c "nohup tensorboard --logdir=/data/logs --host=0.0.0.0 --port=6006 > /var/log/tensorboard.log 2>&1 &"
fi

echo "ML services started"
EOF

chmod +x /opt/scripts/start_ml_services.sh

# Create systemd service for ML services
cat > /etc/systemd/system/ml-services.service << 'EOF'
[Unit]
Description=ML Development Services
After=multi-user.target gpu-monitor.service

[Service]
Type=forking
ExecStart=/opt/scripts/start_ml_services.sh
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF

systemctl enable ml-services

# Final system configuration
echo "Performing final system configuration..."

# Update GPU memory clock for better performance
nvidia-smi -pm 1  # Enable persistence mode
nvidia-smi -ac 877,1380  # Set memory and graphics clocks (adjust based on GPU model)

# Configure system limits for ML workloads
cat >> /etc/security/limits.conf << 'EOF'
mluser soft nofile 1048576
mluser hard nofile 1048576
mluser soft nproc 1048576
mluser hard nproc 1048576
EOF

# Reboot to ensure all drivers are loaded properly
echo "GPU cluster node initialization complete. Rebooting..."
reboot