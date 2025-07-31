# FusionAlpha Setup Guide

## System Requirements

### Hardware
- **CPU**: 4+ cores recommended
- **RAM**: 16GB minimum, 32GB recommended
- **GPU**: NVIDIA GPU with CUDA support (optional but recommended for BICEP acceleration)
- **Storage**: 50GB free space for data and models

### Software
- **OS**: Linux (Ubuntu 20.04+), macOS 10.15+, or Windows 10+
- **Python**: 3.8, 3.9, or 3.10
- **CUDA**: 11.7+ (if using GPU)

## Installation Steps

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/FusionAlpha.git
cd FusionAlpha

# Create and activate virtual environment
python -m venv venv

# On Linux/macOS:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 2. Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install core dependencies
pip install -r requirements.txt

# For development
pip install -e .

# For GPU support (optional)
pip install cupy-cuda11x triton
```

### 3. Download Pre-trained Models

```bash
# Create artifacts directory if it doesn't exist
mkdir -p artifacts

# Download FinBERT model (will be cached automatically on first use)
python -c "from transformers import AutoModel; AutoModel.from_pretrained('yiyanghkust/finbert-tone')"
```

### 4. Configure API Keys (Optional)

Create a `.env` file in the project root:

```env
# Market Data API (optional - system supports free data sources)
ALPHA_VANTAGE_API_KEY=your_key_here
POLYGON_API_KEY=your_key_here

# News API (optional)
NEWS_API_KEY=your_key_here

# Trading API (for live trading only)
ALPACA_API_KEY=your_key_here
ALPACA_SECRET_KEY=your_secret_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets  # Use paper trading first!
```

### 5. Verify Installation

```bash
# Run tests
pytest tests/

# Quick functionality check
python -c "import fusion_alpha; print('FusionAlpha successfully installed!')"
```

## Data Setup

### 1. Historical Market Data

```bash
# Download sample S&P 500 data
python data_collection/sp500_collector.py --start-date 2023-01-01 --end-date 2023-12-31

# Download news data
python data_collection/free_news_collector.py --days 30
```

### 2. Training Data Preparation

```bash
# Prepare training dataset
python -m fusion_alpha.pipelines.prepare_dataset

# This will create:
# - training_data/processed_data.parquet
# - training_data/contradiction_labels.csv
```

## Model Training

### 1. Train Contradiction Detection Model

```bash
# Train with default parameters
python -m fusion_alpha.pipelines.train_contradiction_model

# Train with custom config
python -m fusion_alpha.pipelines.train_contradiction_model \
    --epochs 50 \
    --batch-size 32 \
    --learning-rate 0.0001
```

### 2. Train FusionNet

```bash
# Train the main trading model
python -m fusion_alpha.pipelines.train_fusion

# Hyperparameter tuning (optional)
python -m fusion_alpha.pipelines.tune_fusion --n-trials 100
```

## Running the System

### 1. Backtesting

```bash
# Run backtest on historical data
python -m fusion_alpha.pipelines.run_pipeline \
    --start-date 2023-01-01 \
    --end-date 2023-12-31 \
    --initial-capital 100000

# Results will be saved to backtesting/results/
```

### 2. Paper Trading

```bash
# Start paper trading simulation
python paper_trading/simulation.py

# Monitor performance
python paper_trading/simulation.py --mode monitor
```

### 3. Live Trading API

```bash
# Start the FastAPI server
uvicorn fusion_alpha.routers.live_trading:app --reload

# API will be available at http://localhost:8000
# Documentation at http://localhost:8000/docs
```

## Troubleshooting

### Common Issues

1. **CUDA out of memory**
   ```bash
   # Reduce batch size in config
   export BATCH_SIZE=16
   ```

2. **Missing dependencies**
   ```bash
   # Install system dependencies
   sudo apt-get install python3-dev build-essential  # Linux
   brew install python3-dev  # macOS
   ```

3. **API rate limits**
   - Use free data sources (yfinance)
   - Implement caching for API calls
   - Use paper trading for testing

### Performance Optimization

1. **Enable GPU acceleration**
   ```python
   # In config/integrated_pipeline_config.py
   DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
   ```

2. **Parallel data processing**
   ```bash
   # Set number of workers
   export NUM_WORKERS=4
   ```

3. **Memory optimization**
   - Use data generators instead of loading all data
   - Enable gradient checkpointing for large models

## Next Steps

1. Read the [Theory Documentation](docs/theory.md)
2. Explore [Example Notebooks](examples/)
3. Run the [Tutorial](examples/01_getting_started.ipynb)
4. Join our [Discord Community](https://discord.gg/fusionalpha)

## Support

- GitHub Issues: https://github.com/yourusername/FusionAlpha/issues
- Documentation: https://fusionalpha.readthedocs.io
- Email: support@fusionalpha.ai