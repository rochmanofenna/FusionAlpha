#!/usr/bin/env python3
"""
FusionAlpha - Unified Trading Pipeline

Complete BICEP + ENN + FusionAlpha integration with monitoring infrastructure.
Main entry point for the unified trading system.
"""

import sys
import os
import argparse
import logging
from datetime import datetime
import asyncio

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
    ║  • Contradiction Detection Engine                      ║
    ║  • GPU-Accelerated Computation                        ║
    ║  • Real-time Monitoring Dashboard                     ║
    ║  • Complete Infrastructure Stack                      ║
    ║                                                        ║
    ║  Dashboard: http://localhost:5000                     ║
    ║  Monitor:   ws://localhost:8765                       ║
    ╚════════════════════════════════════════════════════════╝
    """
    print(banner)

async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='FusionAlpha Trading Pipeline')
    parser.add_argument('--mode', choices=['web', 'monitor', 'backtest'], 
                       default='web', help='Run mode')
    parser.add_argument('--tickers', nargs='+', 
                       default=['AAPL', 'GOOGL', 'NVDA', 'TSLA', 'MSFT'],
                       help='Tickers to monitor')
    
    args = parser.parse_args()
    
    print_banner()
    logger.info(f"🚀 Starting FusionAlpha in {args.mode} mode")
    
    if args.mode == 'web':
        # Run web dashboard
        logger.info("Starting web dashboard...")
        from app import main as run_app
        run_app()
        
    elif args.mode == 'monitor':
        # Run monitoring only
        logger.info("Starting monitoring dashboard...")
        from infrastructure.enhanced_monitoring import PipelineMonitor
        monitor = PipelineMonitor()
        monitor.start()
        
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("Stopping monitor...")
            monitor.stop()
            
    elif args.mode == 'backtest':
        # Run backtest
        logger.info("Running backtest analysis...")
        # Import and run backtest logic here
        
    logger.info("FusionAlpha shutdown complete")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 FusionAlpha stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)