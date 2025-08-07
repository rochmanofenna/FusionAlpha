#!/usr/bin/env python3
"""
Underhype Pipeline Deployment Module

Provides deployment functionality for the complete underhype detection system.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.unified_pipeline_integration import UnifiedPipelineIntegration
from infrastructure.enhanced_monitoring import PipelineMonitor
import logging

logger = logging.getLogger(__name__)

class UnderhypeDeploymentPipeline:
    """Deployment manager for the unified pipeline"""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.pipeline = None
        self.monitor = None
        
    def deploy(self):
        """Deploy the complete pipeline"""
        try:
            # Initialize unified pipeline
            self.pipeline = UnifiedPipelineIntegration(self.config)
            
            # Initialize monitoring
            self.monitor = PipelineMonitor()
            
            logger.info("Pipeline deployed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            return False
    
    def start_monitoring(self):
        """Start the monitoring system"""
        if self.monitor:
            self.monitor.start()
            return True
        return False
    
    def get_status(self):
        """Get deployment status"""
        return {
            'pipeline_deployed': self.pipeline is not None,
            'monitoring_active': self.monitor is not None,
            'status': 'deployed' if self.pipeline else 'not_deployed'
        }