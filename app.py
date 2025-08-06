#!/usr/bin/env python3
"""
UnderhypePipeline Flask Web Application

RESTful API server for the complete FusionAlpha + BICEP + ENN pipeline.
Provides web interface and API endpoints for underhype signal detection.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json

# Import pipeline components
from scripts.deploy_underhype_pipeline import UnderhypeDeploymentPipeline
from config.underhype_config import get_production_config, PRIORITY_TICKERS
from run_underhype_only import CompleteUnderhypePipeline

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global pipeline instance
pipeline_instance = None
pipeline_status = {
    'initialized': False,
    'running': False,
    'last_update': None,
    'error': None
}

# Results storage
live_results = {
    'signals': [],
    'performance': {},
    'status': 'stopped'
}

def initialize_pipeline():
    """Initialize the pipeline in a thread-safe manner"""
    global pipeline_instance, pipeline_status
    
    try:
        logger.info("🚀 Initializing UnderhypePipeline...")
        config = get_production_config()
        pipeline_instance = UnderhypeDeploymentPipeline(config)
        
        pipeline_status.update({
            'initialized': True,
            'running': False,
            'last_update': datetime.now().isoformat(),
            'error': None
        })
        
        logger.info("✅ Pipeline initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Pipeline initialization failed: {e}")
        pipeline_status.update({
            'initialized': False,
            'running': False,
            'last_update': datetime.now().isoformat(),
            'error': str(e)
        })
        return False

# HTML Templates
DASHBOARD_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>UnderhypePipeline Dashboard</title>
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
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 UnderhypePipeline Dashboard</h1>
            <p>Complete FusionAlpha + BICEP + ENN Integration</p>
            <p><strong>Focus:</strong> Underhype Signals Only (100% Historical Success Rate)</p>
        </div>
        
        <div class="card">
            <h2>Pipeline Status</h2>
            <div id="status" class="status stopped">
                Initializing...
            </div>
            <div class="controls">
                <button onclick="startPipeline()" class="btn-success">Start Detection</button>
                <button onclick="stopPipeline()" class="btn-danger">Stop Detection</button>
                <button onclick="refreshStatus()" class="btn-info">Refresh Status</button>
                <button onclick="runBacktest()" class="btn-primary">Run Backtest</button>
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
                    <div class="metric-value" id="uptime">0h</div>
                    <div class="metric-label">Uptime</div>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h2>Recent Underhype Signals</h2>
            <div id="signals" class="signals">
                <p>No signals detected yet. Start the pipeline to begin detection.</p>
            </div>
        </div>
    </div>

    <script>
        function startPipeline() {
            fetch('/api/start', {method: 'POST'})
                .then(response => response.json())
                .then(data => {
                    updateStatus();
                    if (data.success) {
                        alert('Pipeline started successfully!');
                    } else {
                        alert('Failed to start pipeline: ' + data.error);
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
        
        function runBacktest() {
            const startDate = prompt('Enter start date (YYYY-MM-DD):', '2023-01-01');
            const endDate = prompt('Enter end date (YYYY-MM-DD):', '2024-01-01');
            
            if (startDate && endDate) {
                fetch('/api/backtest', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({start_date: startDate, end_date: endDate})
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        alert(`Backtest completed: ${data.results.performance_summary.total_signals} signals generated`);
                        updateStatus();
                    } else {
                        alert('Backtest failed: ' + data.error);
                    }
                })
                .catch(error => alert('Error: ' + error));
            }
        }
        
        function refreshStatus() {
            updateStatus();
        }
        
        function updateStatus() {
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    const statusEl = document.getElementById('status');
                    const status = data.pipeline_status;
                    
                    if (status.running) {
                        statusEl.className = 'status running';
                        statusEl.innerHTML = '🟢 Pipeline Running - Detecting underhype signals...';
                    } else if (status.initialized) {
                        statusEl.className = 'status stopped';
                        statusEl.innerHTML = '🔴 Pipeline Stopped - Ready to start detection';
                    } else {
                        statusEl.className = 'status initializing';
                        statusEl.innerHTML = '🟡 Pipeline Initializing...';
                    }
                    
                    if (status.error) {
                        statusEl.innerHTML += '<br>❌ Error: ' + status.error;
                    }
                    
                    // Update metrics
                    const signals = data.signals || [];
                    document.getElementById('signal-count').textContent = signals.length;
                    
                    if (data.performance) {
                        document.getElementById('signals-per-hour').textContent = 
                            (data.performance.signals_per_hour || 0).toFixed(1);
                        document.getElementById('avg-confidence').textContent = 
                            (data.performance.avg_confidence || 0).toFixed(2);
                        document.getElementById('uptime').textContent = 
                            Math.floor(data.performance.uptime || 0) + 'h';
                    }
                    
                    // Update signals
                    updateSignals(signals);
                })
                .catch(error => console.error('Status update failed:', error));
        }
        
        function updateSignals(signals) {
            const signalsEl = document.getElementById('signals');
            
            if (signals.length === 0) {
                signalsEl.innerHTML = '<p>No signals detected yet.</p>';
                return;
            }
            
            const signalsHtml = signals.slice(-10).reverse().map(signal => `
                <div class="signal ${signal.signal_strength || 'medium'}">
                    <strong>${signal.ticker}</strong> - ${signal.date}
                    <div>Confidence: ${signal.confidence.toFixed(2)} | 
                         Sentiment: ${signal.sentiment_score.toFixed(3)} | 
                         Price Movement: ${(signal.price_movement * 100).toFixed(2)}%</div>
                    <div>Signal: <strong>BUY</strong> (Underhype Opportunity)</div>
                    <div style="font-size: 0.9em; color: #666; margin-top: 5px;">
                        ${signal.headline.substring(0, 80)}...
                    </div>
                </div>
            `).join('');
            
            signalsEl.innerHTML = signalsHtml;
        }
        
        // Auto-refresh every 30 seconds
        setInterval(updateStatus, 30000);
        
        // Initial status update
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
    global pipeline_instance, pipeline_status, live_results
    
    try:
        # Get pipeline status
        status_data = {
            'pipeline_status': pipeline_status.copy(),
            'signals': live_results.get('signals', []),
            'performance': live_results.get('performance', {}),
            'timestamp': datetime.now().isoformat()
        }
        
        # Add detailed status if pipeline is initialized
        if pipeline_instance:
            try:
                detailed_status = pipeline_instance.get_pipeline_status()
                status_data['detailed_status'] = detailed_status
                status_data['performance'].update({
                    'uptime': detailed_status.get('pipeline_uptime', 0),
                })
            except Exception as e:
                logger.error(f"Failed to get detailed status: {e}")
        
        return jsonify(status_data)
        
    except Exception as e:
        logger.error(f"Status API error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/start', methods=['POST'])
def api_start():
    """Start pipeline detection"""
    global pipeline_instance, pipeline_status, live_results
    
    try:
        if not pipeline_status['initialized']:
            if not initialize_pipeline():
                return jsonify({'success': False, 'error': 'Pipeline initialization failed'}), 500
        
        if pipeline_status['running']:
            return jsonify({'success': False, 'error': 'Pipeline already running'})
        
        # Start detection in background thread
        def run_detection():
            global live_results, pipeline_status
            
            pipeline_status['running'] = True
            live_results['status'] = 'running'
            
            try:
                # Run detection for 24 hours by default
                results = pipeline_instance.run_live_detection(duration_hours=24)
                
                # Update live results
                live_results.update({
                    'signals': results.get('signals_generated', []),
                    'performance': results.get('performance_metrics', {}),
                    'status': 'completed'
                })
                
            except Exception as e:
                logger.error(f"Detection failed: {e}")
                live_results['status'] = f'error: {str(e)}'
            finally:
                pipeline_status['running'] = False
        
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
    global pipeline_status, live_results
    
    try:
        pipeline_status['running'] = False
        live_results['status'] = 'stopped'
        
        return jsonify({'success': True, 'message': 'Pipeline stopped'})
        
    except Exception as e:
        logger.error(f"Stop API error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/backtest', methods=['POST'])
def api_backtest():
    """Run backtest validation"""
    global pipeline_instance
    
    try:
        if not pipeline_status['initialized']:
            if not initialize_pipeline():
                return jsonify({'success': False, 'error': 'Pipeline initialization failed'}), 500
        
        data = request.get_json()
        start_date = data.get('start_date', '2023-01-01')
        end_date = data.get('end_date', '2024-01-01')
        
        # Run backtest
        results = pipeline_instance.run_backtest_validation(start_date, end_date)
        
        return jsonify({
            'success': True, 
            'results': results,
            'message': f'Backtest completed: {len(results.get("signals_generated", []))} signals'
        })
        
    except Exception as e:
        logger.error(f"Backtest API error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/signals')
def api_signals():
    """Get recent signals"""
    global live_results
    
    try:
        limit = request.args.get('limit', 50, type=int)
        signals = live_results.get('signals', [])[-limit:]
        
        return jsonify({
            'signals': signals,
            'total': len(signals),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Signals API error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/config')
def api_config():
    """Get pipeline configuration"""
    try:
        config = get_production_config()
        return jsonify({
            'config': {
                'confidence_threshold': config.config.confidence_threshold,
                'max_portfolio_allocation': config.config.max_portfolio_allocation,
                'base_position_size': config.config.base_position_size,
                'priority_tickers': PRIORITY_TICKERS,
                'enable_bicep': config.config.enable_bicep,
                'enable_enn': config.config.enable_enn,
                'device': config.config.device
            }
        })
        
    except Exception as e:
        logger.error(f"Config API error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'pipeline_initialized': pipeline_status['initialized'],
        'pipeline_running': pipeline_status['running']
    })

def main():
    """Main application entry point"""
    
    print("🚀 UNDERHYPE PIPELINE WEB SERVER")
    print("=" * 50)
    print("Starting Flask web application...")
    print("Dashboard will be available at: http://localhost:5000")
    print("API endpoints available at: http://localhost:5000/api/*")
    print("=" * 50)
    
    # Initialize pipeline on startup
    initialize_pipeline()
    
    # Start Flask app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,
        threaded=True
    )

if __name__ == '__main__':
    main()