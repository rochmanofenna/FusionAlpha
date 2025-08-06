#!/usr/bin/env python3
"""
Main Unified Pipeline Runner

Complete integration of BICEP + ENN + FusionAlpha with monitoring.
This is the main entry point for running the full pipeline.
"""

import sys
import os
import argparse
import logging
import signal
import time
from datetime import datetime
import json
import asyncio
from typing import Dict, List, Optional

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import pipeline components
from core.unified_pipeline_integration import UnifiedPipelineIntegration
from infrastructure.enhanced_monitoring import PipelineMonitor
from config.underhype_config import get_production_config
from data.market_data_manager import MarketDataManager
from app import app as flask_app
import threading

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class UnifiedPipelineOrchestrator:
    """
    Main orchestrator for the unified pipeline system
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._load_config()
        self.running = False
        
        # Initialize components
        logger.info("🚀 Initializing Unified Pipeline Orchestrator...")
        
        # Core pipeline
        self.pipeline = UnifiedPipelineIntegration(self.config.get('pipeline', {}))
        
        # Monitoring
        self.monitor = PipelineMonitor(self.config.get('monitoring', {}))
        
        # Market data manager
        self.market_data = MarketDataManager()
        
        # Signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("✅ Pipeline orchestrator initialized successfully")
    
    def _load_config(self) -> Dict:
        """Load configuration from file or defaults"""
        config_path = os.path.join(os.path.dirname(__file__), 'config', 'unified_config.json')
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        
        # Default configuration
        return {
            'pipeline': {
                'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                'enable_bicep': True,
                'enable_enn': True,
                'enable_graph': True,
                'bicep': {
                    'n_paths': 100,
                    'n_steps': 50,
                    'scenarios_per_ticker': 20
                },
                'enn': {
                    'num_neurons': 128,
                    'num_states': 8,
                    'entanglement_dim': 16,
                    'memory_length': 10,
                    'dropout_rate': 0.1
                },
                'graph': {
                    'node_dim': 64,
                    'edge_dim': 32,
                    'hidden_dim': 128,
                    'num_layers': 3
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
                'websocket_port': 8765,
                'alert_thresholds': {
                    'gpu_utilization': 95,
                    'memory_percent': 90,
                    'temperature': 85,
                    'latency': 100
                }
            },
            'data': {
                'tickers': ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA', 'META', 'AMZN', 'ORCL', 'AVGO'],
                'update_interval': 60,  # seconds
                'lookback_days': 90
            },
            'execution': {
                'mode': 'simulation',  # 'simulation' or 'live'
                'max_positions': 10,
                'rebalance_interval': 3600  # seconds
            }
        }
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info("🛑 Shutdown signal received")
        self.stop()
    
    def start(self):
        """Start the unified pipeline"""
        if self.running:
            logger.warning("Pipeline already running")
            return
        
        logger.info("🚀 Starting Unified Pipeline...")
        self.running = True
        
        # Start monitoring
        self.monitor.start()
        logger.info("✅ Monitoring started")
        
        # Start web interface in background
        if self.config.get('enable_web_interface', True):
            web_thread = threading.Thread(target=self._run_web_interface)
            web_thread.daemon = True
            web_thread.start()
            logger.info("✅ Web interface started on http://localhost:5000")
        
        # Start main processing loop
        asyncio.run(self._main_loop())
    
    def stop(self):
        """Stop the unified pipeline"""
        logger.info("🛑 Stopping Unified Pipeline...")
        self.running = False
        
        # Stop monitoring
        self.monitor.stop()
        
        # Save final state
        self._save_state()
        
        logger.info("✅ Pipeline stopped successfully")
    
    async def _main_loop(self):
        """Main processing loop"""
        logger.info("📊 Entering main processing loop")
        
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
                
                # Rebalance portfolio
                if current_time - last_rebalance > self.config['execution']['rebalance_interval']:
                    await self._rebalance_portfolio(signals)
                    last_rebalance = current_time
                
                # Short sleep to prevent CPU spinning
                await asyncio.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                await asyncio.sleep(5.0)  # Wait before retrying
    
    async def _update_market_data(self):
        """Update market data for all tickers"""
        logger.info("📈 Updating market data...")
        
        tickers = self.config['data']['tickers']
        updated_count = 0
        
        for ticker in tickers:
            try:
                # Fetch latest data
                data = self.market_data.get_latest_data(ticker)
                if data:
                    updated_count += 1
            except Exception as e:
                logger.error(f"Failed to update {ticker}: {e}")
        
        logger.info(f"✅ Updated {updated_count}/{len(tickers)} tickers")
    
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
                        'graph_data': None  # Would be populated with actual graph data
                    }
                    
                    # Process through pipeline
                    signal = self.pipeline.process_market_data(market_data)
                    
                    # Filter for high-confidence signals
                    if signal.confidence > 0.7 and signal.direction == 'buy':
                        signals.append({
                            'ticker': signal.ticker,
                            'timestamp': signal.timestamp,
                            'signal_type': signal.signal_type,
                            'confidence': signal.confidence,
                            'final_size': signal.final_size,
                            'expected_return': signal.expected_return,
                            'headline': signal.headline
                        })
                        
                        logger.info(f"📊 Signal detected: {ticker} - {signal.signal_type} "
                                  f"(conf: {signal.confidence:.2f})")
            
            except Exception as e:
                logger.error(f"Error processing {ticker}: {e}")
        
        return signals
    
    async def _rebalance_portfolio(self, signals: List[Dict]):
        """Rebalance portfolio based on signals"""
        if not signals:
            return
        
        logger.info(f"🔄 Rebalancing portfolio with {len(signals)} signals")
        
        # Sort by confidence
        signals.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Take top signals up to max positions
        max_positions = self.config['execution']['max_positions']
        selected_signals = signals[:max_positions]
        
        for signal in selected_signals:
            logger.info(f"  → {signal['ticker']}: {signal['signal_type']} "
                       f"size={signal['final_size']:.4f} "
                       f"exp_ret={signal['expected_return']:.2f}%")
        
        # In live mode, this would execute trades
        if self.config['execution']['mode'] == 'live':
            logger.warning("Live execution not implemented - running in simulation mode")
    
    def _run_web_interface(self):
        """Run Flask web interface"""
        flask_app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
    
    def _save_state(self):
        """Save current state to file"""
        state = {
            'timestamp': datetime.now().isoformat(),
            'config': self.config,
            'metrics': self.monitor.get_dashboard_data()
        }
        
        state_file = os.path.join(os.path.dirname(__file__), 'pipeline_state.json')
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"State saved to {state_file}")

def print_banner():
    """Print startup banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║             UNIFIED TRADING PIPELINE SYSTEM                  ║
    ║                                                              ║
    ║  BICEP + ENN + FusionAlpha Integration                      ║
    ║  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━                      ║
    ║                                                              ║
    ║  Components:                                                 ║
    ║  • BICEP: GPU-accelerated stochastic computation            ║
    ║  • ENN: Entangled Neural Networks with memory               ║
    ║  • FusionAlpha: Contradiction detection engine              ║
    ║  • Risk Management: Limit/colimit based controls            ║
    ║                                                              ║
    ║  Monitoring: http://localhost:5000                           ║
    ║  WebSocket:  ws://localhost:8765                            ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Unified Trading Pipeline')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--mode', choices=['simulation', 'live'], default='simulation',
                       help='Execution mode')
    parser.add_argument('--no-web', action='store_true', help='Disable web interface')
    parser.add_argument('--tickers', nargs='+', help='Override ticker list')
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # Load config
    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # Override with command line args
    if args.mode:
        config.setdefault('execution', {})['mode'] = args.mode
    if args.no_web:
        config['enable_web_interface'] = False
    if args.tickers:
        config.setdefault('data', {})['tickers'] = args.tickers
    
    # Create and start orchestrator
    orchestrator = UnifiedPipelineOrchestrator(config)
    
    try:
        orchestrator.start()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
    finally:
        orchestrator.stop()

if __name__ == "__main__":
    # Required imports for runtime
    import torch
    
    # Run main
    main()