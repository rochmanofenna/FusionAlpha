#!/usr/bin/env python3
"""
Enhanced Monitoring Infrastructure for Unified Pipeline

Real-time monitoring dashboard with comprehensive metrics for:
- BICEP GPU computation performance
- ENN state evolution and memory
- FusionAlpha signal generation
- Risk management and portfolio exposure
"""

import time
import json
import numpy as np
import pandas as pd
from collections import deque, defaultdict
from datetime import datetime, timedelta
import threading
import queue
import logging
from typing import Dict, List, Optional, Any
import asyncio
import websockets

# GPU monitoring
try:
    import pynvml
    pynvml.nvmlInit()
    GPU_MONITORING = True
except:
    GPU_MONITORING = False

# Prometheus metrics (optional)
try:
    from prometheus_client import Counter, Gauge, Histogram, Summary
    PROMETHEUS_AVAILABLE = True
except:
    PROMETHEUS_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MetricsCollector:
    """Centralized metrics collection for all pipeline components"""
    
    def __init__(self, retention_minutes: int = 60):
        self.retention_minutes = retention_minutes
        self.metrics_store = defaultdict(lambda: deque(maxlen=retention_minutes * 60))
        self.lock = threading.Lock()
        
        # Define metric schemas
        self.metric_schemas = {
            'bicep': {
                'gpu_utilization': {'unit': '%', 'type': 'gauge'},
                'memory_usage': {'unit': 'MB', 'type': 'gauge'},
                'kernel_throughput': {'unit': 'ops/sec', 'type': 'counter'},
                'sde_latency': {'unit': 'ms', 'type': 'histogram'},
                'temperature': {'unit': '°C', 'type': 'gauge'}
            },
            'enn': {
                'active_neurons': {'unit': 'count', 'type': 'gauge'},
                'entanglement_strength': {'unit': 'score', 'type': 'gauge'},
                'state_memory': {'unit': 'MB', 'type': 'gauge'},
                'forward_pass_time': {'unit': 'ms', 'type': 'histogram'},
                'sparsity_ratio': {'unit': '%', 'type': 'gauge'}
            },
            'fusion_alpha': {
                'signals_generated': {'unit': 'count', 'type': 'counter'},
                'underhype_detected': {'unit': 'count', 'type': 'counter'},
                'overhype_detected': {'unit': 'count', 'type': 'counter'},
                'contradiction_rate': {'unit': '%', 'type': 'gauge'},
                'confidence_avg': {'unit': 'score', 'type': 'gauge'}
            },
            'risk': {
                'portfolio_exposure': {'unit': '%', 'type': 'gauge'},
                'current_leverage': {'unit': 'x', 'type': 'gauge'},
                'var_95': {'unit': '$', 'type': 'gauge'},
                'sharpe_ratio': {'unit': 'ratio', 'type': 'gauge'},
                'max_drawdown': {'unit': '%', 'type': 'gauge'}
            },
            'system': {
                'cpu_usage': {'unit': '%', 'type': 'gauge'},
                'memory_total': {'unit': 'GB', 'type': 'gauge'},
                'disk_io': {'unit': 'MB/s', 'type': 'gauge'},
                'network_latency': {'unit': 'ms', 'type': 'gauge'},
                'uptime': {'unit': 'hours', 'type': 'counter'}
            }
        }
        
        # Initialize Prometheus metrics if available
        if PROMETHEUS_AVAILABLE:
            self._init_prometheus_metrics()
    
    def _init_prometheus_metrics(self):
        """Initialize Prometheus metric collectors"""
        self.prom_metrics = {}
        
        for component, metrics in self.metric_schemas.items():
            for metric_name, config in metrics.items():
                full_name = f"pipeline_{component}_{metric_name}"
                
                if config['type'] == 'counter':
                    self.prom_metrics[full_name] = Counter(
                        full_name, f"{component} {metric_name}"
                    )
                elif config['type'] == 'gauge':
                    self.prom_metrics[full_name] = Gauge(
                        full_name, f"{component} {metric_name}"
                    )
                elif config['type'] == 'histogram':
                    self.prom_metrics[full_name] = Histogram(
                        full_name, f"{component} {metric_name}"
                    )
    
    def record_metric(self, component: str, metric: str, value: float, 
                     timestamp: Optional[datetime] = None):
        """Record a metric value"""
        if timestamp is None:
            timestamp = datetime.now()
        
        with self.lock:
            key = f"{component}.{metric}"
            self.metrics_store[key].append({
                'value': value,
                'timestamp': timestamp.isoformat()
            })
            
            # Update Prometheus if available
            if PROMETHEUS_AVAILABLE:
                prom_key = f"pipeline_{component}_{metric}"
                if prom_key in self.prom_metrics:
                    prom_metric = self.prom_metrics[prom_key]
                    if hasattr(prom_metric, 'set'):
                        prom_metric.set(value)
                    elif hasattr(prom_metric, 'inc'):
                        prom_metric.inc(value)
                    elif hasattr(prom_metric, 'observe'):
                        prom_metric.observe(value)
    
    def get_metrics(self, component: str, metric: str, 
                   last_minutes: Optional[int] = None) -> List[Dict]:
        """Get metric history"""
        with self.lock:
            key = f"{component}.{metric}"
            data = list(self.metrics_store.get(key, []))
            
            if last_minutes:
                cutoff = datetime.now() - timedelta(minutes=last_minutes)
                data = [d for d in data 
                       if datetime.fromisoformat(d['timestamp']) > cutoff]
            
            return data
    
    def get_latest_metrics(self) -> Dict[str, Any]:
        """Get latest values for all metrics"""
        latest = {}
        
        with self.lock:
            for key, values in self.metrics_store.items():
                if values:
                    latest[key] = values[-1]['value']
        
        return latest

class GPUMonitor:
    """Enhanced GPU monitoring for BICEP operations"""
    
    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.handle = None
        
        if GPU_MONITORING:
            try:
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
                self.name = pynvml.nvmlDeviceGetName(self.handle).decode('utf-8')
                logger.info(f"GPU Monitor initialized for: {self.name}")
            except Exception as e:
                logger.error(f"Failed to initialize GPU monitor: {e}")
                self.handle = None
    
    def get_metrics(self) -> Dict[str, float]:
        """Get current GPU metrics"""
        if not self.handle:
            return {}
        
        try:
            # Utilization
            util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            
            # Memory
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            mem_used_mb = mem_info.used / (1024 * 1024)
            mem_total_mb = mem_info.total / (1024 * 1024)
            
            # Temperature
            temp = pynvml.nvmlDeviceGetTemperature(
                self.handle, pynvml.NVML_TEMPERATURE_GPU
            )
            
            # Power
            try:
                power = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000  # mW to W
            except:
                power = 0
            
            # Clock speeds
            try:
                sm_clock = pynvml.nvmlDeviceGetClockInfo(
                    self.handle, pynvml.NVML_CLOCK_SM
                )
                mem_clock = pynvml.nvmlDeviceGetClockInfo(
                    self.handle, pynvml.NVML_CLOCK_MEM
                )
            except:
                sm_clock = 0
                mem_clock = 0
            
            return {
                'gpu_utilization': util.gpu,
                'memory_usage': mem_used_mb,
                'memory_total': mem_total_mb,
                'memory_percent': (mem_used_mb / mem_total_mb) * 100,
                'temperature': temp,
                'power_usage': power,
                'sm_clock': sm_clock,
                'mem_clock': mem_clock
            }
            
        except Exception as e:
            logger.error(f"Error getting GPU metrics: {e}")
            return {}

class PipelineMonitor:
    """Main monitoring orchestrator for unified pipeline"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._get_default_config()
        self.metrics_collector = MetricsCollector()
        self.gpu_monitor = GPUMonitor() if GPU_MONITORING else None
        
        # Component monitors
        self.monitors = {
            'bicep': BICEPMonitor(self.metrics_collector),
            'enn': ENNMonitor(self.metrics_collector),
            'fusion': FusionMonitor(self.metrics_collector),
            'risk': RiskMonitor(self.metrics_collector)
        }
        
        # Monitoring thread
        self.running = False
        self.monitor_thread = None
        
        # WebSocket server for real-time updates
        self.ws_server = None
        self.ws_clients = set()
    
    def _get_default_config(self) -> Dict:
        """Default monitoring configuration"""
        return {
            'update_interval': 1.0,  # seconds
            'gpu_monitoring': True,
            'websocket_port': 8765,
            'alert_thresholds': {
                'gpu_utilization': 95,  # %
                'memory_percent': 90,   # %
                'temperature': 85,      # °C
                'latency': 100         # ms
            }
        }
    
    def start(self):
        """Start monitoring"""
        if self.running:
            return
        
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        # Start WebSocket server
        asyncio.get_event_loop().run_until_complete(
            self._start_websocket_server()
        )
        
        logger.info("Pipeline monitoring started")
    
    def stop(self):
        """Stop monitoring"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join()
        
        logger.info("Pipeline monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                # Collect GPU metrics
                if self.gpu_monitor:
                    gpu_metrics = self.gpu_monitor.get_metrics()
                    for metric, value in gpu_metrics.items():
                        self.metrics_collector.record_metric('bicep', metric, value)
                
                # Collect component metrics
                for name, monitor in self.monitors.items():
                    monitor.collect_metrics()
                
                # Check alerts
                self._check_alerts()
                
                # Broadcast updates
                self._broadcast_updates()
                
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
            
            time.sleep(self.config['update_interval'])
    
    def _check_alerts(self):
        """Check for alert conditions"""
        latest = self.metrics_collector.get_latest_metrics()
        alerts = []
        
        thresholds = self.config['alert_thresholds']
        
        # GPU alerts
        if 'bicep.gpu_utilization' in latest:
            if latest['bicep.gpu_utilization'] > thresholds['gpu_utilization']:
                alerts.append({
                    'level': 'warning',
                    'component': 'bicep',
                    'message': f"GPU utilization high: {latest['bicep.gpu_utilization']:.1f}%"
                })
        
        if 'bicep.temperature' in latest:
            if latest['bicep.temperature'] > thresholds['temperature']:
                alerts.append({
                    'level': 'critical',
                    'component': 'bicep',
                    'message': f"GPU temperature critical: {latest['bicep.temperature']}°C"
                })
        
        # Process alerts
        for alert in alerts:
            logger.warning(f"ALERT [{alert['level']}] {alert['component']}: {alert['message']}")
    
    async def _start_websocket_server(self):
        """Start WebSocket server for real-time updates"""
        async def handler(websocket, path):
            self.ws_clients.add(websocket)
            try:
                await websocket.wait_closed()
            finally:
                self.ws_clients.remove(websocket)
        
        self.ws_server = await websockets.serve(
            handler, 'localhost', self.config['websocket_port']
        )
    
    def _broadcast_updates(self):
        """Broadcast metrics to WebSocket clients"""
        if not self.ws_clients:
            return
        
        metrics = self.get_dashboard_data()
        message = json.dumps(metrics)
        
        # Send to all connected clients
        asyncio.get_event_loop().run_until_complete(
            asyncio.gather(*[client.send(message) for client in self.ws_clients])
        )
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get current dashboard data"""
        latest = self.metrics_collector.get_latest_metrics()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'components': {
                'bicep': {
                    'status': 'active' if latest.get('bicep.gpu_utilization', 0) > 10 else 'idle',
                    'gpu_utilization': latest.get('bicep.gpu_utilization', 0),
                    'memory_usage': latest.get('bicep.memory_usage', 0),
                    'temperature': latest.get('bicep.temperature', 0),
                    'throughput': latest.get('bicep.kernel_throughput', 0)
                },
                'enn': {
                    'status': 'active',
                    'active_neurons': latest.get('enn.active_neurons', 0),
                    'entanglement': latest.get('enn.entanglement_strength', 0),
                    'sparsity': latest.get('enn.sparsity_ratio', 0)
                },
                'fusion_alpha': {
                    'status': 'active',
                    'signals_total': latest.get('fusion_alpha.signals_generated', 0),
                    'underhype_count': latest.get('fusion_alpha.underhype_detected', 0),
                    'contradiction_rate': latest.get('fusion_alpha.contradiction_rate', 0),
                    'avg_confidence': latest.get('fusion_alpha.confidence_avg', 0)
                },
                'risk': {
                    'portfolio_exposure': latest.get('risk.portfolio_exposure', 0),
                    'current_leverage': latest.get('risk.current_leverage', 1.0),
                    'var_95': latest.get('risk.var_95', 0),
                    'sharpe_ratio': latest.get('risk.sharpe_ratio', 0)
                }
            }
        }

class BICEPMonitor:
    """Monitor BICEP-specific metrics"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self.kernel_calls = 0
        self.last_reset = time.time()
    
    def collect_metrics(self):
        """Collect BICEP metrics"""
        # Simulated metrics - replace with actual BICEP calls
        throughput = np.random.uniform(1e6, 1e7)  # ops/sec
        latency = np.random.uniform(0.5, 2.0)  # ms
        
        self.metrics.record_metric('bicep', 'kernel_throughput', throughput)
        self.metrics.record_metric('bicep', 'sde_latency', latency)

class ENNMonitor:
    """Monitor ENN-specific metrics"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
    
    def collect_metrics(self):
        """Collect ENN metrics"""
        # Simulated metrics - replace with actual ENN state
        active_neurons = np.random.randint(100, 128)
        entanglement = np.random.uniform(0.3, 0.9)
        sparsity = np.random.uniform(0.1, 0.4)
        
        self.metrics.record_metric('enn', 'active_neurons', active_neurons)
        self.metrics.record_metric('enn', 'entanglement_strength', entanglement)
        self.metrics.record_metric('enn', 'sparsity_ratio', sparsity * 100)

class FusionMonitor:
    """Monitor FusionAlpha metrics"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self.signal_count = 0
        self.underhype_count = 0
    
    def collect_metrics(self):
        """Collect FusionAlpha metrics"""
        # Increment counters (simulated)
        if np.random.random() < 0.1:  # 10% chance of signal
            self.signal_count += 1
            if np.random.random() < 0.7:  # 70% are underhype
                self.underhype_count += 1
        
        self.metrics.record_metric('fusion_alpha', 'signals_generated', self.signal_count)
        self.metrics.record_metric('fusion_alpha', 'underhype_detected', self.underhype_count)
        
        # Calculate rates
        if self.signal_count > 0:
            contradiction_rate = (self.underhype_count / self.signal_count) * 100
            self.metrics.record_metric('fusion_alpha', 'contradiction_rate', contradiction_rate)

class RiskMonitor:
    """Monitor risk management metrics"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
    
    def collect_metrics(self):
        """Collect risk metrics"""
        # Simulated portfolio metrics
        exposure = np.random.uniform(10, 25)  # % of portfolio
        leverage = np.random.uniform(1.0, 2.5)
        var_95 = np.random.uniform(1000, 5000)
        sharpe = np.random.uniform(1.0, 3.0)
        
        self.metrics.record_metric('risk', 'portfolio_exposure', exposure)
        self.metrics.record_metric('risk', 'current_leverage', leverage)
        self.metrics.record_metric('risk', 'var_95', var_95)
        self.metrics.record_metric('risk', 'sharpe_ratio', sharpe)

def create_grafana_dashboard() -> Dict:
    """Create Grafana dashboard configuration"""
    return {
        "dashboard": {
            "title": "Unified Pipeline Monitoring",
            "panels": [
                {
                    "title": "GPU Performance",
                    "type": "graph",
                    "targets": [
                        {"expr": "pipeline_bicep_gpu_utilization"},
                        {"expr": "pipeline_bicep_temperature"}
                    ]
                },
                {
                    "title": "Signal Generation Rate",
                    "type": "graph",
                    "targets": [
                        {"expr": "rate(pipeline_fusion_alpha_signals_generated[5m])"}
                    ]
                },
                {
                    "title": "Portfolio Risk",
                    "type": "gauge",
                    "targets": [
                        {"expr": "pipeline_risk_portfolio_exposure"}
                    ]
                }
            ]
        }
    }

# Example usage
if __name__ == "__main__":
    # Start monitoring
    monitor = PipelineMonitor()
    monitor.start()
    
    print("Enhanced Pipeline Monitoring Started")
    print(f"WebSocket updates available at ws://localhost:8765")
    print("Press Ctrl+C to stop...")
    
    try:
        while True:
            # Print dashboard snapshot every 10 seconds
            time.sleep(10)
            data = monitor.get_dashboard_data()
            print(f"\nDashboard Update @ {data['timestamp']}")
            print(json.dumps(data['components'], indent=2))
    
    except KeyboardInterrupt:
        print("\nStopping monitor...")
        monitor.stop()