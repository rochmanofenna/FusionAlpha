#!/usr/bin/env python3
"""
Underhype Pipeline Configuration

Configuration settings for the underhype detection system.
"""

import os

# Priority tickers for underhype detection
PRIORITY_TICKERS = ['AAPL', 'GOOGL', 'NVDA', 'TSLA', 'MSFT', 'AMZN', 'META', 'NFLX', 'ORCL', 'CRM']

def get_production_config():
    """Get production configuration"""
    return {
        'pipeline': {
            'device': 'cuda' if os.environ.get('USE_CUDA', 'true').lower() == 'true' else 'cpu',
            'enable_bicep': True,
            'enable_enn': True,
            'enable_graph': True,
            'bicep': {
                'n_paths': int(os.environ.get('BICEP_PATHS', 100)),
                'n_steps': int(os.environ.get('BICEP_STEPS', 50)),
                'scenarios_per_ticker': 20
            },
            'enn': {
                'num_neurons': int(os.environ.get('ENN_NEURONS', 128)),
                'num_states': int(os.environ.get('ENN_STATES', 8)),
                'entanglement_dim': 16,
                'memory_length': 10,
                'dropout_rate': 0.1
            },
            'risk': {
                'max_leverage': float(os.environ.get('MAX_LEVERAGE', 3.0)),
                'base_position_size': 0.02,
                'volatility_adjustment': True
            }
        },
        'monitoring': {
            'update_interval': 1.0,
            'gpu_monitoring': True,
            'websocket_port': int(os.environ.get('WS_PORT', 8765))
        },
        'data': {
            'tickers': PRIORITY_TICKERS,
            'update_interval': 60,
            'lookback_days': 90
        }
    }

def get_development_config():
    """Get development configuration"""
    config = get_production_config()
    # Override for development
    config['pipeline']['bicep']['n_paths'] = 50
    config['data']['tickers'] = PRIORITY_TICKERS[:5]  # Limit for dev
    return config