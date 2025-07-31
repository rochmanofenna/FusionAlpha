#!/usr/bin/env python3
"""
Training Data Pipeline for Fusion Alpha

This creates the data structure and format needed to train our models.
Includes price data, technical indicators, sentiment, and contradiction labels.
"""

import os
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

# Setup directories - works both locally and on Colab
if os.path.exists("/content"):
    # Running on Colab
    DATA_DIR = Path("/content/mismatch-trading/training_data")
else:
    # Running locally
    DATA_DIR = Path("/home/ryan/trading/mismatch-trading/training_data")

DATA_DIR.mkdir(exist_ok=True)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrainingDataGenerator:
    """Generate synthetic training data for initial model development"""
    
    def __init__(self, num_symbols: int = 50, num_days: int = 252):
        self.num_symbols = num_symbols
        self.num_days = num_days
        self.bars_per_day = 78  # 5-min bars in 6.5 hour trading day
        
        # Create realistic symbol list with sectors
        self.symbols = self._generate_symbols()
        
    def _generate_symbols(self) -> Dict[str, str]:
        """Generate symbol list with sector distribution matching S&P 500"""
        sectors = {
            "Technology": 14,
            "Healthcare": 7,
            "Financials": 7,
            "Consumer Discretionary": 6,
            "Communication": 5,
            "Industrials": 4,
            "Consumer Staples": 3,
            "Energy": 2,
            "Utilities": 1,
            "Real Estate": 1
        }
        
        symbols = {}
        for sector, count in sectors.items():
            for i in range(count):
                # Create realistic ticker symbols
                prefix = sector[:3].upper()
                symbol = f"{prefix}{i:02d}"
                symbols[symbol] = sector
                
        return symbols
    
    def generate_price_data(self, symbol: str, sector: str) -> pd.DataFrame:
        """Generate realistic price data with sector characteristics"""
        
        # Sector-specific parameters
        sector_params = {
            "Technology": {"volatility": 0.03, "trend": 0.0002, "mean_reversion": 0.8},
            "Healthcare": {"volatility": 0.02, "trend": 0.0001, "mean_reversion": 0.9},
            "Financials": {"volatility": 0.025, "trend": 0.00005, "mean_reversion": 0.85},
            "Consumer Discretionary": {"volatility": 0.025, "trend": 0.00015, "mean_reversion": 0.82},
            "Communication": {"volatility": 0.028, "trend": 0.00018, "mean_reversion": 0.83},
            "Industrials": {"volatility": 0.022, "trend": 0.00008, "mean_reversion": 0.87},
            "Consumer Staples": {"volatility": 0.015, "trend": 0.00005, "mean_reversion": 0.95},
            "Energy": {"volatility": 0.035, "trend": -0.00005, "mean_reversion": 0.75},
            "Utilities": {"volatility": 0.012, "trend": 0.00003, "mean_reversion": 0.98},
            "Real Estate": {"volatility": 0.018, "trend": 0.00006, "mean_reversion": 0.92}
        }
        
        params = sector_params.get(sector, {"volatility": 0.02, "trend": 0.0001, "mean_reversion": 0.85})
        
        # Generate timestamps
        timestamps = []
        current_date = datetime.now() - timedelta(days=self.num_days)
        
        for day in range(self.num_days):
            # Market hours: 9:30 AM to 4:00 PM ET
            market_open = current_date.replace(hour=9, minute=30, second=0)
            
            for bar in range(self.bars_per_day):
                timestamp = market_open + timedelta(minutes=5 * bar)
                timestamps.append(timestamp)
                
            current_date += timedelta(days=1)
            # Skip weekends
            if current_date.weekday() >= 5:
                current_date += timedelta(days=2 if current_date.weekday() == 5 else 1)
        
        # Generate price series using geometric Brownian motion
        num_bars = len(timestamps)
        dt = 1 / self.bars_per_day  # Time step
        
        # Initialize
        prices = np.zeros(num_bars)
        prices[0] = 100  # Starting price
        
        # Add regime changes
        regime_changes = np.random.randint(0, num_bars, size=5)
        regimes = np.ones(num_bars)
        for rc in regime_changes:
            regimes[rc:rc+self.bars_per_day*10] *= np.random.choice([0.5, 1.5, 2.0])
        
        # Generate price path
        for i in range(1, num_bars):
            # Mean reversion component
            mean_reversion = params['mean_reversion'] * (100 - prices[i-1]) * dt
            
            # Trend component
            trend = params['trend'] * dt
            
            # Random component with regime adjustment
            random_shock = params['volatility'] * np.sqrt(dt) * np.random.randn() * regimes[i]
            
            # Update price
            prices[i] = prices[i-1] * (1 + trend + random_shock) + mean_reversion
            prices[i] = max(prices[i], 10)  # Floor at $10
        
        # Create OHLCV data
        df = pd.DataFrame({
            'timestamp': timestamps,
            'symbol': symbol,
            'sector': sector
        })
        
        # Generate OHLC from price series with realistic patterns
        df['close'] = prices
        df['open'] = df['close'].shift(1).fillna(prices[0])
        
        # High/Low with realistic spreads
        spread = params['volatility'] * np.random.uniform(0.2, 0.5, num_bars)
        df['high'] = df[['open', 'close']].max(axis=1) * (1 + spread)
        df['low'] = df[['open', 'close']].min(axis=1) * (1 - spread)
        
        # Volume with intraday pattern
        base_volume = 1000000
        intraday_pattern = np.concatenate([
            np.linspace(1.5, 1.0, self.bars_per_day // 3),  # Morning high
            np.linspace(1.0, 0.7, self.bars_per_day // 3),  # Midday low
            np.linspace(0.7, 1.2, self.bars_per_day - 2 * (self.bars_per_day // 3))  # Afternoon rise
        ])
        
        volumes = []
        for i in range(num_bars):
            bar_of_day = i % self.bars_per_day
            pattern_factor = intraday_pattern[min(bar_of_day, len(intraday_pattern)-1)]
            volume = base_volume * pattern_factor * (1 + 0.3 * np.random.randn())
            volumes.append(max(int(volume), 100))
            
        df['volume'] = volumes
        
        return df
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to price data"""
        
        # Price changes
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Moving averages
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        bb_period = 20
        df['bb_middle'] = df['close'].rolling(window=bb_period).mean()
        bb_std = df['close'].rolling(window=bb_period).std()
        df['bb_upper'] = df['bb_middle'] + (2 * bb_std)
        df['bb_lower'] = df['bb_middle'] - (2 * bb_std)
        
        # ATR (Average True Range)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['atr'] = true_range.rolling(window=14).mean()
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Volatility
        df['volatility'] = df['returns'].rolling(window=20).std()
        
        return df
    
    def add_sentiment_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate synthetic sentiment data correlated with price movements"""
        
        # Base sentiment from price momentum with lag
        momentum = df['returns'].rolling(window=12).mean().shift(2)  # 1-hour momentum with lag
        
        # Add noise and sector-specific bias
        sector = df['sector'].iloc[0]
        sector_bias = {
            "Technology": 0.1,  # Generally positive sentiment
            "Energy": -0.05,    # Generally negative sentiment
            "Utilities": 0.0,   # Neutral
        }.get(sector, 0.0)
        
        # Generate sentiment score
        df['sentiment_score'] = momentum * 5 + sector_bias + np.random.normal(0, 0.2, len(df))
        df['sentiment_score'] = df['sentiment_score'].clip(-1, 1)
        
        # Sentiment volume (number of mentions/articles)
        df['sentiment_volume'] = np.random.poisson(10, len(df))
        
        # Sentiment volatility
        df['sentiment_volatility'] = df['sentiment_score'].rolling(window=20).std()
        
        # News events (random spikes)
        news_events = np.random.random(len(df)) < 0.02  # 2% chance of news event
        df.loc[news_events, 'sentiment_volume'] *= 5
        df.loc[news_events, 'sentiment_score'] *= 1.5
        
        return df
    
    def create_contradiction_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create labels for contradiction detection"""
        
        # Future returns for different horizons
        df['future_return_5m'] = df['close'].shift(-1) / df['close'] - 1
        df['future_return_1h'] = df['close'].shift(-12) / df['close'] - 1
        df['future_return_1d'] = df['close'].shift(-78) / df['close'] - 1
        
        # Direction labels
        df['direction_5m'] = (df['future_return_5m'] > 0).astype(int)
        df['direction_1h'] = (df['future_return_1h'] > 0).astype(int)
        
        # Contradiction detection
        df['contradiction_type'] = 'none'
        
        # Overhype: Strong positive sentiment but negative returns
        overhype_mask = (df['sentiment_score'] > 0.5) & (df['future_return_1h'] < -0.002)
        df.loc[overhype_mask, 'contradiction_type'] = 'overhype'
        
        # Underhype: Strong negative sentiment but positive returns
        underhype_mask = (df['sentiment_score'] < -0.5) & (df['future_return_1h'] > 0.002)
        df.loc[underhype_mask, 'contradiction_type'] = 'underhype'
        
        # Paradox: High volatility but no price movement
        paradox_mask = (df['volatility'] > df['volatility'].quantile(0.8)) & (np.abs(df['future_return_1h']) < 0.0005)
        df.loc[paradox_mask, 'contradiction_type'] = 'paradox'
        
        # Add contradiction strength
        df['contradiction_strength'] = 0.0
        df.loc[df['contradiction_type'] != 'none', 'contradiction_strength'] = \
            np.abs(df.loc[df['contradiction_type'] != 'none', 'sentiment_score'] * 
                   df.loc[df['contradiction_type'] != 'none', 'future_return_1h'])
        
        return df
    
    def generate_training_dataset(self) -> pd.DataFrame:
        """Generate complete training dataset"""
        
        all_data = []
        
        logger.info(f"Generating training data for {self.num_symbols} symbols...")
        
        for symbol, sector in self.symbols.items():
            # Generate price data
            df = self.generate_price_data(symbol, sector)
            
            # Add technical indicators
            df = self.add_technical_indicators(df)
            
            # Add sentiment data
            df = self.add_sentiment_data(df)
            
            # Create labels
            df = self.create_contradiction_labels(df)
            
            # Drop NaN rows from calculations
            df = df.dropna()
            
            all_data.append(df)
            logger.info(f"Generated {len(df)} samples for {symbol} ({sector})")
        
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Add global features
        combined_df['hour'] = combined_df['timestamp'].dt.hour
        combined_df['minute'] = combined_df['timestamp'].dt.minute
        combined_df['day_of_week'] = combined_df['timestamp'].dt.dayofweek
        
        return combined_df
    
    def save_training_data(self, df: pd.DataFrame):
        """Save training data in format ready for model training"""
        
        # Save full dataset
        output_path = DATA_DIR / "synthetic_training_data.parquet"
        df.to_parquet(output_path, engine='pyarrow')
        logger.info(f"Saved training data to {output_path}")
        
        # Save metadata
        metadata = {
            "generation_date": datetime.now().isoformat(),
            "num_symbols": self.num_symbols,
            "num_days": self.num_days,
            "total_samples": len(df),
            "features": {
                "price": ["open", "high", "low", "close", "volume"],
                "technical": ["sma_20", "sma_50", "ema_12", "ema_26", "macd", "rsi", "bb_upper", "bb_lower", "atr", "volatility"],
                "sentiment": ["sentiment_score", "sentiment_volume", "sentiment_volatility"],
                "labels": ["future_return_5m", "future_return_1h", "direction_5m", "direction_1h", "contradiction_type", "contradiction_strength"]
            },
            "contradiction_distribution": df['contradiction_type'].value_counts().to_dict(),
            "sector_distribution": df['sector'].value_counts().to_dict()
        }
        
        metadata_path = DATA_DIR / "training_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Print summary
        print("\nTraining Data Summary")
        print("="*50)
        print(f"Total samples: {len(df):,}")
        print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"Number of symbols: {df['symbol'].nunique()}")
        
        print("\nContradiction distribution:")
        for ctype, count in df['contradiction_type'].value_counts().items():
            print(f"  {ctype}: {count:,} ({count/len(df)*100:.1f}%)")
        
        print("\nFeature columns:", len(df.columns))
        print("Ready for training!")
        
        return df

def create_sample_batch(df: pd.DataFrame, batch_size: int = 32) -> Dict:
    """Create a sample batch in the format expected by the pipeline"""
    
    # Sample random batch
    batch_df = df.sample(n=batch_size)
    
    # Extract features for pipeline
    batch = {
        # Price features (10 dim)
        'tech_features': batch_df[['rsi', 'macd', 'bb_upper', 'bb_lower', 'volume_ratio',
                                   'volatility', 'atr', 'returns', 'volume_ratio', 'hour']].values,
        
        # Sentiment features (3 dim)
        'sentiment_scores': batch_df['sentiment_score'].values,
        
        # Price movements for contradiction detection
        'price_movements': batch_df['returns'].values,
        
        # Labels
        'targets': batch_df['future_return_5m'].values,
        'contradiction_labels': batch_df['contradiction_type'].values
    }
    
    return batch

def main():
    """Generate synthetic training data"""
    
    print("Generating Synthetic Training Data")
    print("="*50)
    
    # Create generator
    generator = TrainingDataGenerator(
        num_symbols=50,  # 50 synthetic symbols
        num_days=60      # 60 trading days (about 3 months)
    )
    
    # Generate dataset
    df = generator.generate_training_dataset()
    
    # Save dataset
    generator.save_training_data(df)
    
    # Create sample batch
    print("\nSample batch for testing:")
    batch = create_sample_batch(df)
    print(f"Tech features shape: {batch['tech_features'].shape}")
    print(f"Sentiment shape: {batch['sentiment_scores'].shape}")
    print(f"Unique contradictions in batch: {np.unique(batch['contradiction_labels'])}")
    
    print("\nTraining data ready!")
    print(f"Location: {DATA_DIR}")
    print("\nNext steps:")
    print("1. Create training loop")
    print("2. Train Fusion Alpha model")
    print("3. Build backtesting framework")

if __name__ == "__main__":
    main()