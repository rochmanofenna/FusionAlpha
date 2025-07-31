# FusionAlpha Project Structure

## Core Components

### 1. Contradiction Detection (`fusion_alpha/`)
The heart of the system - detects divergences between sentiment and price action.

- **`contradiction_model.py`**: Neural network for contradiction classification
- **`pipelines/contradiction_engine.py`**: Adaptive threshold-based detection
- **`models/finbert.py`**: Financial sentiment analysis using FinBERT

### 2. Event Neural Network (`enn/`)
Temporal pattern recognition and event sequence modeling.

- **`model.py`**: Main ENN architecture
- **`memory.py`**: Dynamic memory management for market states
- **`event_processing.py`**: Real-time event stream processing

### 3. BICEP Backend (`backends/bicep/`)
Brownian motion-based price path simulation.

- **`brownian_motion.py`**: Core stochastic process implementation
- **`stochastic_control.py`**: Optimal control for entry/exit points
- **`sde_int/`**: GPU-accelerated SDE integration (C++/CUDA)

### 4. Trading System (`fusion_alpha/routers/`)
Live trading implementation and API endpoints.

- **`live_trading.py`**: Main trading loop and order execution
- **`option_screener.py`**: Options strategy selection
- **`predict_router.py`**: Real-time prediction API

### 5. Data Collection (`data_collection/`)
Market data and news gathering infrastructure.

- **`free_market_data.py`**: Yahoo Finance integration
- **`free_news_collector.py`**: RSS/API news aggregation
- **`sp500_collector.py`**: S&P 500 universe data

### 6. Backtesting (`backtesting/`)
Historical performance validation.

- **`comprehensive_backtest.py`**: Full strategy backtesting
- **`results/`**: Performance metrics and reports

### 7. Paper Trading (`paper_trading/`)
Risk-free live testing environment.

- **`simulation.py`**: Real-time paper trading engine

## Key Files

### Configuration
- **`requirements.txt`**: All Python dependencies with versions
- **`setup.py`**: Package installation script
- **`.gitignore`**: Repository cleanliness

### Documentation
- **`README.md`**: Project overview and theory
- **`SETUP.md`**: Detailed installation guide
- **`LICENSE`**: MIT license

### Examples
- **`examples/contradiction_demo.py`**: Visual demonstration of theory
- **`examples/quick_start.py`**: Basic usage example

## Data Flow

1. **Input**: Market data + News feeds
2. **Processing**: 
   - FinBERT → Sentiment scores
   - Technical analysis → Price features
3. **Detection**: Contradiction engine identifies divergences
4. **Prediction**: ENN + FusionNet generate signals
5. **Execution**: Trading router places orders
6. **Validation**: Backtesting verifies performance

## Training Pipeline

1. **Data Preparation**: `prepare_dataset.py`
2. **Model Training**: 
   - `train_contradiction_model.py`
   - `train_fusion.py`
3. **Hyperparameter Tuning**: `tune_fusion.py`
4. **Cross-validation**: `kfold_cross_validation.py`
5. **Walk-forward Testing**: `walk_forward_validation.py`

## Testing

- **Unit Tests**: `tests/` directory
- **Integration Tests**: `test_*_integration.py`
- **Performance Benchmarks**: `performance_benchmarks.py`

## Models

Pre-trained models in `artifacts/`:
- `contradiction_model.pth`: Contradiction classifier
- `fusion_net_best_weights.pth`: Main trading model
- `optuna_study.db`: Hyperparameter optimization results