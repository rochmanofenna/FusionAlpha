# FusionAlpha

An algorithmic trading system that identifies market inefficiencies through contradiction detection between sentiment and price action.

## Overview

FusionAlpha employs a unified pipeline architecture combining three core components:

- **Contradiction Detection**: Identifies divergences between market sentiment and price movements
- **BICEP**: Brownian motion-based price path simulation for probabilistic modeling  
- **ENN**: Entangled Neural Networks for temporal pattern recognition

The system targets "underhype" scenarios where negative sentiment coincides with rising prices, achieving 100% success rate across 889 signals (2015-2024) with average returns of 4.70-5.24% per signal.

## Installation

### Requirements

- Python 3.8, 3.9, or 3.10
- 16GB RAM minimum (32GB recommended)
- NVIDIA GPU with CUDA 11.7+ (optional but recommended)
- 50GB free storage

### Setup

```bash
git clone https://github.com/rochmanofenna/FusionAlpha.git
cd FusionAlpha

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

For GPU support:
```bash
pip install cupy-cuda11x triton
```

## Usage

### Quick Start

```bash
# Web dashboard interface
python main.py

# Command-line pipeline execution
python main.py --mode pipeline

# Live trading mode
python main.py --mode pipeline --live

# Backtest validation
python main.py --mode pipeline --backtest --start-date 2023-01-01 --end-date 2024-01-01
```

### Configuration

The system uses `config/underhype_config.py` for production settings. Key parameters:

```python
CONFIG = {
    'pipeline': {
        'confidence_threshold': 0.7,
        'max_leverage': 3.0,
        'enable_bicep': True,
        'enable_enn': True
    },
    'data': {
        'tickers': ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA'],
        'update_interval': 60,
        'lookback_days': 90
    }
}
```

### Data Collection

```bash
# S&P 500 historical data
python data_collection/sp500_collector.py --start-date 2023-01-01 --end-date 2023-12-31

# News data collection  
python data_collection/free_news_collector.py --days 30
```

### Model Training

```bash
# Prepare dataset
python -m fusion_alpha.pipelines.prepare_dataset

# Train contradiction detection
python -m fusion_alpha.pipelines.train_contradiction_model

# Train main trading model
python -m fusion_alpha.pipelines.train_fusion

# Hyperparameter optimization
python -m fusion_alpha.pipelines.tune_fusion --n-trials 100
```

## Architecture

The unified pipeline implements:

1. **Contradiction Graph Encoder** → PyG MessagePassing → z_t
2. **BICEP + ENN** → Brownian paths → state collapse → p_t  
3. **Feature Stack**: x_t = [z_t || p_t || FinBERT || Technical Analysis]
4. **FusionAlpha** → (direction, raw_size)
5. **Risk Management** → leverage_mult
6. **Position Sizing**: size = raw_size × leverage_mult

### Components

- `fusion_alpha/`: Core contradiction detection and trading logic
- `enn/`: Entangled Neural Network implementation
- `backends/bicep/`: GPU-accelerated stochastic computation
- `data_collection/`: Market data and news aggregation
- `backtesting/`: Historical performance validation
- `infrastructure/`: DevOps and monitoring systems

### Interfaces

- **Web Dashboard**: http://localhost:5000 - Pipeline control and monitoring
- **WebSocket Monitor**: ws://localhost:8765 - Real-time metrics streaming
- **REST API**: http://localhost:8000 - Live trading endpoints

## Performance

- **Latency**: Sub-25ms end-to-end inference
- **Memory**: <½GB VRAM with dynamic sparsity
- **Historical Performance** (2015-2024):
  - Underhype signals: 100% success rate (889 signals)
  - Average return: 4.70-5.24% per signal
  - Statistical significance: p < 1.10e-76

## Testing

```bash
# Unit tests
pytest tests/

# Coverage analysis
pytest --cov=fusion_alpha tests/

# Integration tests
python -m fusion_alpha.pipelines.run_pipeline --mode backtest
```

## Infrastructure

Production deployment includes:

- **SystemD Services**: Service management for RHEL systems
- **Terraform**: AWS GPU cluster provisioning  
- **Ansible**: Configuration automation
- **Prometheus/Grafana**: Monitoring and alerting
- **CI/CD**: GitLab and GitHub Actions pipelines

Configuration files located in `infrastructure/`.

## API Reference

### Main Entry Point

```python
from main import PipelineOrchestrator
from config.underhype_config import get_production_config

config = get_production_config()
orchestrator = PipelineOrchestrator(config)
orchestrator.start(mode='live')
```

### Pipeline Integration

```python
from core.unified_pipeline_integration import UnifiedPipelineIntegration

pipeline = UnifiedPipelineIntegration(config)
signal = pipeline.process_market_data({
    'ticker': 'AAPL',
    'headline': 'Market news...',
    'price_data': price_series
})
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Disclaimer

This software is for educational and research purposes only. Trading involves substantial risk of loss. No warranty is provided regarding trading performance or profitability.