#!/usr/bin/env python3
"""
Complete Underhype Pipeline Runner

Runs the complete underhype detection pipeline with all components.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.unified_pipeline_integration import UnifiedPipelineIntegration
from infrastructure.enhanced_monitoring import PipelineMonitor
from config.underhype_config import get_production_config
import logging
import threading
import time
from typing import Dict, List

logger = logging.getLogger(__name__)

class CompleteUnderhypePipeline:
    """Complete underhype detection pipeline"""
    
    def __init__(self, config=None):
        self.config = config or get_production_config()
        self.pipeline = UnifiedPipelineIntegration(self.config['pipeline'])
        self.monitor = PipelineMonitor(self.config['monitoring'])
        self.running = False
        self.signals = []
        
    def start(self):
        """Start the complete pipeline"""
        if self.running:
            return True
            
        try:
            # Start monitoring
            self.monitor.start()
            
            self.running = True
            logger.info("Complete underhype pipeline started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start pipeline: {e}")
            return False
    
    def stop(self):
        """Stop the pipeline"""
        self.running = False
        self.monitor.stop()
        logger.info("Pipeline stopped")
    
    def process_market_data(self, market_data):
        """Process market data through the pipeline"""
        if not self.running:
            return None
            
        try:
            signal = self.pipeline.process_market_data(market_data)
            
            # Only keep underhype signals
            if signal.signal_type == 'underhype' and signal.confidence > 0.7:
                self.signals.append({
                    'ticker': signal.ticker,
                    'timestamp': signal.timestamp.isoformat(),
                    'confidence': signal.confidence,
                    'expected_return': signal.expected_return,
                    'final_size': signal.final_size,
                    'headline': signal.headline
                })
                
                logger.info(f"Underhype signal: {signal.ticker} (conf: {signal.confidence:.2f})")
                return signal
                
        except Exception as e:
            logger.error(f"Error processing market data: {e}")
            
        return None
    
    def get_signals(self):
        """Get recent signals"""
        return self.signals[-10:]  # Return last 10 signals
    
    def get_status(self):
        """Get pipeline status"""
        return {
            'running': self.running,
            'total_signals': len(self.signals),
            'recent_signals': len([s for s in self.signals[-10:] if s]),
            'monitoring_active': self.monitor is not None
        }