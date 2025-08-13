#!/usr/bin/env python3
"""
FusionAlpha - Unified Trading Pipeline System

Main entry point for the complete BICEP + ENN + FusionAlpha integration.
Provides both web interface and pipeline execution capabilities.
"""

import sys
import os
import argparse
import logging
import signal
import threading
import asyncio
import time
import json
from datetime import datetime, timedelta
from typing import Dict, Optional, List

# Use proper package imports instead of sys.path manipulation
# current_dir = os.path.dirname(os.path.abspath(__file__))
# if current_dir not in sys.path:
#     sys.path.insert(0, current_dir)

# Flask imports
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS

# Pipeline imports
from core.unified_pipeline_integration import UnifiedPipelineIntegration
from infrastructure.enhanced_monitoring import PipelineMonitor
from config.underhype_config import get_production_config
from data_collection.free_market_data import FreeMarketDataCollector as MarketDataManager
from scripts.deploy_underhype_pipeline import UnderhypeDeploymentPipeline

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global state
pipeline_state = {
    'initialized': False,
    'running': False,
    'last_update': None,
    'error': None,
    'signals': [],
    'performance': {},
    'pipeline_instance': None,
    'monitor_instance': None
}

class PipelineOrchestrator:
    """
    Main orchestrator for the unified pipeline system
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._load_config()
        self.running = False
        self.signals = []
        self.start_time = None
        
        # Initialize components
        logger.info("Initializing Pipeline Orchestrator...")
        
        # Core pipeline
        self.pipeline = UnifiedPipelineIntegration(self.config.get('pipeline', {}))
        
        # Monitoring
        self.monitor = PipelineMonitor(self.config.get('monitoring', {}))
        
        # Market data manager
        self.market_data = MarketDataManager()
        
        # Deployment pipeline for underhype
        prod_config = get_production_config()
        self.underhype_pipeline = UnderhypeDeploymentPipeline(prod_config)
        
        # Signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("Pipeline orchestrator initialized successfully")
    
    def _load_config(self) -> Dict:
        """Load configuration from file or defaults"""
        config_path = os.path.join(os.path.dirname(__file__), 'config', 'unified_config.json')
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        
        # Default configuration
        return {
            'pipeline': {
                'enable_bicep': True,
                'enable_enn': True,
                'enable_graph': True,
                'confidence_threshold': 0.7,
                'bicep': {
                    'n_paths': 100,
                    'n_steps': 50,
                    'scenarios_per_ticker': 20
                },
                'enn': {
                    'num_neurons': 128,
                    'num_states': 8,
                    'entanglement_dim': 16,
                    'memory_length': 10
                },
                'risk': {
                    'max_leverage': 3.0,
                    'base_position_size': 0.02,
                    'volatility_adjustment': True
                }
            },
            'monitoring': {
                'update_interval': 1.0,
                'gpu_monitoring': True,
                'websocket_port': 8765
            },
            'data': {
                'tickers': ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA', 'META', 'AMZN'],
                'update_interval': 60,
                'lookback_days': 90
            },
            'execution': {
                'mode': 'simulation',
                'max_positions': 10,
                'rebalance_interval': 3600
            }
        }
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info("Shutdown signal received")
        self.stop()
    
    def start(self, mode: str = 'live'):
        """Start the pipeline in specified mode"""
        if self.running:
            logger.warning("Pipeline already running")
            return
        
        logger.info(f"Starting Pipeline in {mode} mode...")
        self.running = True
        self.start_time = datetime.now()
        
        # Start monitoring
        self.monitor.start()
        logger.info("Monitoring started on ws://localhost:8765")
        
        if mode == 'live':
            # Run live detection
            asyncio.run(self._run_live_mode())
        elif mode == 'backtest':
            # Run backtest
            self._run_backtest_mode()
        elif mode == 'underhype':
            # Run underhype-only mode
            self._run_underhype_mode()
    
    def stop(self):
        """Stop the pipeline"""
        logger.info("Stopping Pipeline...")
        self.running = False
        
        # Stop monitoring
        self.monitor.stop()
        
        # Save final state
        self._save_state()
        
        # Print summary
        self._print_summary()
        
        logger.info("Pipeline stopped successfully")
    
    async def _run_live_mode(self):
        """Run pipeline in live mode"""
        logger.info("Running in LIVE mode - processing real-time market data")
        
        last_update = time.time()
        last_rebalance = time.time()
        
        while self.running:
            try:
                current_time = time.time()
                
                # Update market data
                if current_time - last_update > self.config['data']['update_interval']:
                    await self._update_market_data()
                    last_update = current_time
                
                # Process signals
                signals = await self._process_signals()
                
                # Log high-confidence signals
                for signal in signals:
                    if signal['confidence'] > 0.8:
                        logger.info(f"HIGH CONFIDENCE SIGNAL: {signal['ticker']} - "
                                  f"{signal['signal_type']} (conf: {signal['confidence']:.2f})")
                
                # Rebalance portfolio
                if current_time - last_rebalance > self.config['execution']['rebalance_interval']:
                    await self._rebalance_portfolio(signals)
                    last_rebalance = current_time
                
                # Short sleep
                await asyncio.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Error in live mode: {e}", exc_info=True)
                await asyncio.sleep(5.0)
    
    def _run_backtest_mode(self):
        """Run pipeline in backtest mode"""
        logger.info("Running in BACKTEST mode")
        
        # Get backtest parameters
        start_date = self.config.get('backtest', {}).get('start_date', '2023-01-01')
        end_date = self.config.get('backtest', {}).get('end_date', '2024-01-01')
        
        logger.info(f"Backtest period: {start_date} to {end_date}")
        
        try:
            # Run backtest
            results = self.underhype_pipeline.run_backtest_validation(start_date, end_date)
            
            # Process results
            signals_generated = results.get('signals_generated', [])
            performance = results.get('performance_summary', {})
            
            logger.info(f"Backtest completed:")
            logger.info(f"  - Total signals: {len(signals_generated)}")
            logger.info(f"  - Win rate: {performance.get('win_rate', 0):.2%}")
            logger.info(f"  - Average return: {performance.get('avg_return', 0):.2%}")
            logger.info(f"  - Sharpe ratio: {performance.get('sharpe_ratio', 0):.2f}")
            
            # Save results
            self.signals = signals_generated
            
        except Exception as e:
            logger.error(f"Backtest failed: {e}", exc_info=True)
    
    def _run_underhype_mode(self):
        """Run pipeline in underhype-only mode"""
        logger.info("Running in UNDERHYPE mode - focusing on underhype signals only")
        
        try:
            # Run underhype detection
            results = self.underhype_pipeline.run_live_detection(duration_hours=24)
            
            # Process results
            signals = results.get('signals_generated', [])
            self.signals.extend(signals)
            
            logger.info(f"Underhype detection completed: {len(signals)} signals generated")
            
        except Exception as e:
            logger.error(f"Underhype mode failed: {e}", exc_info=True)
    
    async def _update_market_data(self):
        """Update market data for all tickers"""
        logger.debug("Updating market data...")
        
        tickers = self.config['data']['tickers']
        updated_count = 0
        
        for ticker in tickers:
            try:
                data = self.market_data.get_latest_data(ticker)
                if data:
                    updated_count += 1
            except Exception as e:
                logger.error(f"Failed to update {ticker}: {e}")
        
        logger.info(f"Updated {updated_count}/{len(tickers)} tickers")
    
    async def _process_signals(self) -> List[Dict]:
        """Process signals through unified pipeline"""
        signals = []
        
        for ticker in self.config['data']['tickers']:
            try:
                # Get market data
                price_data = self.market_data.get_ticker_data(ticker)
                if price_data is None:
                    continue
                
                # Get latest news
                news = self.market_data.get_latest_news(ticker)
                if not news:
                    continue
                
                # Process each news item
                for news_item in news[:3]:  # Limit to 3 most recent
                    market_data = {
                        'ticker': ticker,
                        'headline': news_item['headline'],
                        'price_data': price_data,
                        'graph_data': None
                    }
                    
                    # Process through pipeline
                    signal = self.pipeline.process_market_data(market_data)
                    
                    # Filter for high-confidence signals
                    if signal.confidence > self.config['pipeline']['confidence_threshold']:
                        signal_dict = {
                            'ticker': signal.ticker,
                            'timestamp': signal.timestamp.isoformat(),
                            'signal_type': signal.signal_type,
                            'confidence': signal.confidence,
                            'final_size': signal.final_size,
                            'expected_return': signal.expected_return,
                            'headline': signal.headline
                        }
                        signals.append(signal_dict)
                        self.signals.append(signal_dict)
            
            except Exception as e:
                logger.error(f"Error processing {ticker}: {e}")
        
        return signals
    
    async def _rebalance_portfolio(self, signals: List[Dict]):
        """Rebalance portfolio based on signals"""
        if not signals:
            return
        
        logger.info(f"Rebalancing portfolio with {len(signals)} signals")
        
        # Sort by confidence
        signals.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Take top signals
        max_positions = self.config['execution']['max_positions']
        selected_signals = signals[:max_positions]
        
        for signal in selected_signals:
            logger.info(f"  - {signal['ticker']}: {signal['signal_type']} "
                       f"size={signal['final_size']:.4f} "
                       f"exp_ret={signal['expected_return']:.2f}%")
        
        if self.config['execution']['mode'] == 'live':
            logger.warning("Live execution not implemented - running in simulation mode")
    
    def _save_state(self):
        """Save current state to file"""
        state = {
            'timestamp': datetime.now().isoformat(),
            'runtime': str(datetime.now() - self.start_time) if self.start_time else None,
            'total_signals': len(self.signals),
            'config': self.config,
            'signals': self.signals[-100:]  # Save last 100 signals
        }
        
        state_file = os.path.join(os.path.dirname(__file__), 'pipeline_state.json')
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"State saved to {state_file}")
    
    def _print_summary(self):
        """Print pipeline execution summary"""
        if self.start_time:
            runtime = datetime.now() - self.start_time
            logger.info(f"\nPipeline Summary:")
            logger.info(f"  - Runtime: {runtime}")
            logger.info(f"  - Total signals: {len(self.signals)}")
            
            if self.signals:
                # Calculate statistics
                high_conf_signals = [s for s in self.signals if s['confidence'] > 0.8]
                avg_confidence = sum(s['confidence'] for s in self.signals) / len(self.signals)
                
                logger.info(f"  - High confidence signals: {len(high_conf_signals)}")
                logger.info(f"  - Average confidence: {avg_confidence:.2f}")

def print_banner():
    """Print FusionAlpha banner"""
    banner = """
    ╔════════════════════════════════════════════════════════╗
    ║                     FUSION ALPHA                       ║
    ║            Unified Trading Pipeline System             ║
    ║                                                        ║
    ║  BICEP + ENN + FusionAlpha Integration                ║
    ║  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━              ║
    ║                                                        ║
    ║  - Contradiction Detection Engine                      ║
    ║  - GPU-Accelerated Computation (BICEP)               ║
    ║  - Entangled Neural Networks (ENN)                   ║
    ║  - Real-time Monitoring Dashboard                     ║
    ║                                                        ║
    ║  Web Dashboard: http://localhost:5000                 ║
    ║  Monitoring:    ws://localhost:8765                   ║
    ╚════════════════════════════════════════════════════════╝
    """
    print(banner)

# HTML Dashboard Template
DASHBOARD_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>FusionAlpha Trading Pipeline</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { text-align: center; margin-bottom: 30px; }
        .card { background: white; padding: 20px; margin: 10px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .status { padding: 10px; border-radius: 4px; margin: 10px 0; }
        .status.running { background: #d4edda; color: #155724; }
        .status.stopped { background: #f8d7da; color: #721c24; }
        .status.initializing { background: #fff3cd; color: #856404; }
        .signals { max-height: 400px; overflow-y: auto; }
        .signal { border: 1px solid #ddd; padding: 10px; margin: 5px 0; border-radius: 4px; }
        .signal.high { border-left: 4px solid #28a745; }
        .signal.medium { border-left: 4px solid #ffc107; }
        .signal.low { border-left: 4px solid #6c757d; }
        .controls button { padding: 10px 20px; margin: 5px; border: none; border-radius: 4px; cursor: pointer; }
        .btn-primary { background: #007bff; color: white; }
        .btn-success { background: #28a745; color: white; }
        .btn-danger { background: #dc3545; color: white; }
        .btn-info { background: #17a2b8; color: white; }
        .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }
        .metric { text-align: center; }
        .metric-value { font-size: 2em; font-weight: bold; color: #007bff; }
        .metric-label { color: #666; font-size: 0.9em; }
        .components { display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin-top: 15px; }
        .component { text-align: center; padding: 15px; border: 1px solid #ddd; border-radius: 4px; }
        .component.active { border-color: #28a745; background: #f8fff9; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>FusionAlpha Trading Pipeline</h1>
            <p>Complete BICEP + ENN + FusionAlpha Integration</p>
        </div>
        
        <div class="card">
            <h2>Pipeline Status</h2>
            <div id="status" class="status stopped">
                System Ready - Click Initialize to begin
            </div>
            <div class="controls">
                <button onclick="initializePipeline()" class="btn-primary">Initialize Pipeline</button>
                <button onclick="startPipeline()" class="btn-success">Start Detection</button>
                <button onclick="stopPipeline()" class="btn-danger">Stop Detection</button>
                <button onclick="refreshStatus()" class="btn-info">Refresh Status</button>
            </div>
        </div>
        
        <div class="card">
            <h2>System Components</h2>
            <div class="components">
                <div id="bicep-status" class="component">
                    <h3>BICEP</h3>
                    <p>GPU-Accelerated Computation</p>
                    <div id="bicep-state">Not Initialized</div>
                </div>
                <div id="enn-status" class="component">
                    <h3>ENN</h3>
                    <p>Entangled Neural Networks</p>
                    <div id="enn-state">Not Initialized</div>
                </div>
                <div id="fusion-status" class="component">
                    <h3>FusionAlpha</h3>
                    <p>Contradiction Detection</p>
                    <div id="fusion-state">Not Initialized</div>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h2>Performance Metrics</h2>
            <div id="metrics" class="metrics">
                <div class="metric">
                    <div class="metric-value" id="signal-count">0</div>
                    <div class="metric-label">Total Signals</div>
                </div>
                <div class="metric">
                    <div class="metric-value" id="signals-per-hour">0.0</div>
                    <div class="metric-label">Signals/Hour</div>
                </div>
                <div class="metric">
                    <div class="metric-value" id="avg-confidence">0.0</div>
                    <div class="metric-label">Avg Confidence</div>
                </div>
                <div class="metric">
                    <div class="metric-value" id="active-positions">0</div>
                    <div class="metric-label">Active Positions</div>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h2>Recent Trading Signals</h2>
            <div id="signals" class="signals">
                <p>No signals detected yet. Initialize and start the pipeline to begin detection.</p>
            </div>
        </div>
    </div>

    <script>
        let autoRefresh = null;
        
        function initializePipeline() {
            fetch('/api/initialize', {method: 'POST'})
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        updateStatus();
                        alert('Pipeline initialized successfully!');
                        startAutoRefresh();
                    } else {
                        alert('Failed to initialize: ' + data.error);
                    }
                })
                .catch(error => alert('Error: ' + error));
        }
        
        function startPipeline() {
            fetch('/api/start', {method: 'POST'})
                .then(response => response.json())
                .then(data => {
                    updateStatus();
                    if (data.success) {
                        alert('Pipeline started successfully!');
                    } else {
                        alert('Failed to start: ' + data.error);
                    }
                })
                .catch(error => alert('Error: ' + error));
        }
        
        function stopPipeline() {
            fetch('/api/stop', {method: 'POST'})
                .then(response => response.json())
                .then(data => {
                    updateStatus();
                    alert('Pipeline stopped');
                })
                .catch(error => alert('Error: ' + error));
        }
        
        function refreshStatus() {
            updateStatus();
        }
        
        function updateStatus() {
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    // Update main status
                    const statusEl = document.getElementById('status');
                    if (data.running) {
                        statusEl.className = 'status running';
                        statusEl.innerHTML = 'Pipeline Running - Detecting trading signals...';
                    } else if (data.initialized) {
                        statusEl.className = 'status stopped';
                        statusEl.innerHTML = 'Pipeline Ready - Click Start to begin detection';
                    } else {
                        statusEl.className = 'status initializing';
                        statusEl.innerHTML = 'Pipeline Not Initialized - Click Initialize to setup';
                    }
                    
                    // Update component status
                    updateComponentStatus('bicep', data.components?.bicep);
                    updateComponentStatus('enn', data.components?.enn);
                    updateComponentStatus('fusion', data.components?.fusion_alpha);
                    
                    // Update metrics
                    document.getElementById('signal-count').textContent = data.signal_count || 0;
                    document.getElementById('signals-per-hour').textContent = 
                        (data.performance?.signals_per_hour || 0).toFixed(1);
                    document.getElementById('avg-confidence').textContent = 
                        (data.performance?.avg_confidence || 0).toFixed(2);
                    document.getElementById('active-positions').textContent = 
                        data.performance?.active_positions || 0;
                    
                    // Update signals
                    updateSignals(data.signals || []);
                })
                .catch(error => console.error('Status update failed:', error));
        }
        
        function updateComponentStatus(component, active) {
            const el = document.getElementById(component + '-status');
            const stateEl = document.getElementById(component + '-state');
            if (active) {
                el.classList.add('active');
                stateEl.textContent = 'Active';
                stateEl.style.color = '#28a745';
            } else {
                el.classList.remove('active');
                stateEl.textContent = 'Inactive';
                stateEl.style.color = '#6c757d';
            }
        }
        
        function updateSignals(signals) {
            const signalsEl = document.getElementById('signals');
            
            if (signals.length === 0) {
                signalsEl.innerHTML = '<p>No signals detected yet.</p>';
                return;
            }
            
            const signalsHtml = signals.slice(-10).reverse().map(signal => {
                const confidenceClass = signal.confidence > 0.8 ? 'high' : 
                                      signal.confidence > 0.6 ? 'medium' : 'low';
                return `
                    <div class="signal ${confidenceClass}">
                        <strong>${signal.ticker}</strong> - ${signal.timestamp}
                        <div>
                            Type: ${signal.signal_type} | 
                            Confidence: ${(signal.confidence * 100).toFixed(1)}% | 
                            Expected Return: ${signal.expected_return?.toFixed(2)}%
                        </div>
                        <div style="font-size: 0.9em; color: #666; margin-top: 5px;">
                            ${signal.headline?.substring(0, 100)}...
                        </div>
                    </div>
                `;
            }).join('');
            
            signalsEl.innerHTML = signalsHtml;
        }
        
        function startAutoRefresh() {
            if (autoRefresh) clearInterval(autoRefresh);
            autoRefresh = setInterval(updateStatus, 5000);
        }
        
        // Initial update
        updateStatus();
    </script>
</body>
</html>
"""

# API Routes
@app.route('/')
def dashboard():
    """Main dashboard view"""
    return render_template_string(DASHBOARD_TEMPLATE)

@app.route('/api/status')
def api_status():
    """Get pipeline status and recent results"""
    global pipeline_state
    
    try:
        status_data = {
            'initialized': pipeline_state['initialized'],
            'running': pipeline_state['running'],
            'timestamp': datetime.now().isoformat(),
            'signals': pipeline_state.get('signals', [])[-10:],  # Last 10 signals
            'signal_count': len(pipeline_state.get('signals', [])),
            'performance': pipeline_state.get('performance', {}),
            'components': {
                'bicep': pipeline_state.get('initialized', False),
                'enn': pipeline_state.get('initialized', False),
                'fusion_alpha': pipeline_state.get('initialized', False)
            }
        }
        
        # Add error info if present
        if pipeline_state.get('error'):
            status_data['error'] = pipeline_state['error']
        
        return jsonify(status_data)
        
    except Exception as e:
        logger.error(f"Status API error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/initialize', methods=['POST'])
def api_initialize():
    """Initialize pipeline components"""
    global pipeline_state
    
    try:
        if pipeline_state['initialized']:
            return jsonify({'success': True, 'message': 'Pipeline already initialized'})
        
        logger.info("Initializing pipeline components...")
        
        # Get production config
        config = get_production_config()
        
        # Initialize unified pipeline
        pipeline_state['pipeline_instance'] = UnderhypeDeploymentPipeline(config)
        
        # Initialize monitor
        pipeline_state['monitor_instance'] = PipelineMonitor()
        
        pipeline_state.update({
            'initialized': True,
            'last_update': datetime.now().isoformat(),
            'error': None
        })
        
        logger.info("Pipeline initialized successfully")
        
        return jsonify({
            'success': True,
            'message': 'Pipeline initialized successfully',
            'components': ['BICEP', 'ENN', 'FusionAlpha']
        })
        
    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        pipeline_state['error'] = str(e)
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/start', methods=['POST'])
def api_start():
    """Start pipeline detection"""
    global pipeline_state
    
    try:
        if not pipeline_state['initialized']:
            return jsonify({'success': False, 'error': 'Pipeline not initialized'})
        
        if pipeline_state['running']:
            return jsonify({'success': False, 'error': 'Pipeline already running'})
        
        # Start detection in background thread
        def run_detection():
            global pipeline_state
            
            pipeline_state['running'] = True
            pipeline_state['monitor_instance'].start()
            
            try:
                # Run live detection
                results = pipeline_state['pipeline_instance'].run_live_detection(duration_hours=24)
                
                # Update results
                pipeline_state['signals'].extend(results.get('signals_generated', []))
                pipeline_state['performance'].update(results.get('performance_metrics', {}))
                
            except Exception as e:
                logger.error(f"Detection failed: {e}")
                pipeline_state['error'] = str(e)
            finally:
                pipeline_state['running'] = False
                pipeline_state['monitor_instance'].stop()
        
        detection_thread = threading.Thread(target=run_detection)
        detection_thread.daemon = True
        detection_thread.start()
        
        return jsonify({'success': True, 'message': 'Pipeline started'})
        
    except Exception as e:
        logger.error(f"Start API error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/stop', methods=['POST'])
def api_stop():
    """Stop pipeline detection"""
    global pipeline_state
    
    try:
        pipeline_state['running'] = False
        
        if pipeline_state.get('monitor_instance'):
            pipeline_state['monitor_instance'].stop()
        
        return jsonify({'success': True, 'message': 'Pipeline stopped'})
        
    except Exception as e:
        logger.error(f"Stop API error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'pipeline_initialized': pipeline_state['initialized'],
        'pipeline_running': pipeline_state['running']
    })

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info("Shutdown signal received")
    if pipeline_state.get('running'):
        pipeline_state['running'] = False
        if pipeline_state.get('monitor_instance'):
            pipeline_state['monitor_instance'].stop()
    sys.exit(0)

def run_pipeline_mode(args):
    """Run pipeline in standalone mode"""
    # Load config
    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # Override with command line args
    if args.tickers:
        config.setdefault('data', {})['tickers'] = args.tickers
    
    if args.duration:
        config.setdefault('execution', {})['duration_hours'] = args.duration
    
    # Create and start orchestrator
    orchestrator = PipelineOrchestrator(config)
    
    try:
        orchestrator.start(mode=args.pipeline_mode)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
    finally:
        orchestrator.stop()

def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(description='FusionAlpha Trading Pipeline')
    parser.add_argument('--mode', choices=['web', 'pipeline', 'monitor'], 
                       default='web', help='Run mode')
    
    # Web mode arguments
    parser.add_argument('--port', type=int, default=5000, help='Web server port')
    parser.add_argument('--host', default='0.0.0.0', help='Web server host')
    
    # Pipeline mode arguments
    parser.add_argument('--pipeline-mode', choices=['live', 'backtest', 'underhype'], 
                       default='live', help='Pipeline execution mode')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--tickers', nargs='+', help='Override ticker list')
    parser.add_argument('--duration', type=int, default=24, 
                       help='Duration in hours (for live/underhype modes)')
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    if args.mode == 'web':
        # Run web interface
        logger.info(f"Starting web dashboard on http://{args.host}:{args.port}")
        app.run(host=args.host, port=args.port, debug=False, threaded=True)
        
    elif args.mode == 'pipeline':
        # Run pipeline directly
        logger.info(f"Starting pipeline in {args.pipeline_mode} mode...")
        run_pipeline_mode(args)
        
    elif args.mode == 'monitor':
        # Run monitor only
        logger.info("Starting monitoring dashboard...")
        monitor = PipelineMonitor()
        monitor.start()
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            monitor.stop()

if __name__ == '__main__':
    main()