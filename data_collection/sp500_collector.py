#!/usr/bin/env python3
"""
S&P 500 Training Data Collection Pipeline

Collects historical data for all S&P 500 stocks including:
- Price data (OHLCV)
- Technical indicators
- News sentiment
- Sector information
"""

import os
import sys
import time
import json
import pickle
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Create data directories
DATA_DIR = Path("/home/ryan/trading/mismatch-trading/training_data")
DATA_DIR.mkdir(exist_ok=True)
(DATA_DIR / "raw").mkdir(exist_ok=True)
(DATA_DIR / "processed").mkdir(exist_ok=True)
(DATA_DIR / "metadata").mkdir(exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(DATA_DIR / 'data_collection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# S&P 500 companies with sectors
SP500_SECTORS = {
    # Technology (28%)
    "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology", 
    "AVGO": "Technology", "ORCL": "Technology", "CSCO": "Technology",
    "ADBE": "Technology", "CRM": "Technology", "ACN": "Technology",
    "INTC": "Technology", "AMD": "Technology", "QCOM": "Technology",
    
    # Healthcare (13%)
    "UNH": "Healthcare", "JNJ": "Healthcare", "LLY": "Healthcare",
    "ABBV": "Healthcare", "PFE": "Healthcare", "MRK": "Healthcare",
    "TMO": "Healthcare", "ABT": "Healthcare", "DHR": "Healthcare",
    
    # Financials (13%)
    "BRK.B": "Financials", "JPM": "Financials", "V": "Financials",
    "MA": "Financials", "BAC": "Financials", "WFC": "Financials",
    "GS": "Financials", "MS": "Financials", "AXP": "Financials",
    
    # Consumer Discretionary (11%)
    "AMZN": "Consumer Discretionary", "TSLA": "Consumer Discretionary",
    "HD": "Consumer Discretionary", "MCD": "Consumer Discretionary",
    "NKE": "Consumer Discretionary", "LOW": "Consumer Discretionary",
    "SBUX": "Consumer Discretionary", "TGT": "Consumer Discretionary",
    
    # Communication Services (9%)
    "GOOGL": "Communication", "GOOG": "Communication", "META": "Communication",
    "NFLX": "Communication", "DIS": "Communication", "CMCSA": "Communication",
    "VZ": "Communication", "T": "Communication", "TMUS": "Communication",
    
    # Industrials (8%)
    "BA": "Industrials", "CAT": "Industrials", "GE": "Industrials",
    "UNP": "Industrials", "HON": "Industrials", "UPS": "Industrials",
    "LMT": "Industrials", "RTX": "Industrials", "DE": "Industrials",
    
    # Consumer Staples (6%)
    "WMT": "Consumer Staples", "PG": "Consumer Staples", "KO": "Consumer Staples",
    "PEP": "Consumer Staples", "COST": "Consumer Staples", "MDLZ": "Consumer Staples",
    
    # Energy (4%)
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy", "SLB": "Energy",
    "EOG": "Energy", "PXD": "Energy", "MPC": "Energy", "VLO": "Energy",
    
    # Others (Utilities, Real Estate, Materials)
    "NEE": "Utilities", "SO": "Utilities", "DUK": "Utilities",
    "AMT": "Real Estate", "PLD": "Real Estate", "CCI": "Real Estate",
    "LIN": "Materials", "APD": "Materials", "SHW": "Materials",
}

class SP500DataCollector:
    """Collects and processes S&P 500 training data"""
    
    def __init__(self, start_date: str = "2021-01-01", end_date: str = None):
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime("%Y-%m-%d")
        self.symbols = list(SP500_SECTORS.keys())
        
        # Data storage
        self.price_data = {}
        self.technical_data = {}
        self.sentiment_data = {}
        
        logger.info(f"Initialized collector for {len(self.symbols)} S&P 500 stocks")
        logger.info(f"Date range: {self.start_date} to {self.end_date}")
        
    def collect_price_data(self, symbol: str) -> pd.DataFrame:
        """Collect historical price data using yfinance"""
        try:
            import yfinance as yf
            
            logger.info(f"Collecting price data for {symbol}")
            
            # Download data
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                start=self.start_date,
                end=self.end_date,
                interval="5m",  # 5-minute bars
                prepost=True,   # Include pre/post market
                actions=False   # Exclude dividends/splits for now
            )
            
            if df.empty:
                logger.warning(f"No data found for {symbol}")
                return pd.DataFrame()
            
            # Clean column names
            df.columns = [col.lower() for col in df.columns]
            
            # Add symbol and sector
            df['symbol'] = symbol
            df['sector'] = SP500_SECTORS.get(symbol, "Unknown")
            
            # Calculate returns
            df['returns_5m'] = df['close'].pct_change()
            df['returns_1h'] = df['close'].pct_change(12)  # 12 * 5min = 1 hour
            df['returns_1d'] = df['close'].pct_change(78)  # 78 * 5min = 6.5 hours (trading day)
            
            # Log data info
            logger.info(f"{symbol}: Collected {len(df)} records from {df.index[0]} to {df.index[-1]}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error collecting {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for the price data"""
        if df.empty:
            return df
            
        try:
            # Simple Moving Averages
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()
            df['sma_200'] = df['close'].rolling(window=200).mean()
            
            # Exponential Moving Averages
            df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
            df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
            
            # MACD
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            df['bb_middle'] = df['close'].rolling(window=20).mean()
            bb_std = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (2 * bb_std)
            df['bb_lower'] = df['bb_middle'] - (2 * bb_std)
            df['bb_width'] = df['bb_upper'] - df['bb_lower']
            df['bb_position'] = (df['close'] - df['bb_lower']) / df['bb_width']
            
            # Volume indicators
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # Price position indicators
            df['high_low_ratio'] = df['high'] / df['low']
            df['close_open_ratio'] = df['close'] / df['open']
            
            # Volatility
            df['volatility_5m'] = df['returns_5m'].rolling(window=20).std()
            df['volatility_1h'] = df['returns_1h'].rolling(window=20).std()
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {str(e)}")
            return df
    
    def collect_sentiment_data(self, symbol: str, df: pd.DataFrame) -> pd.DataFrame:
        """Mock sentiment data collection (would use real news API in production)"""
        if df.empty:
            return df
            
        # For now, generate synthetic sentiment based on price movements
        # In production, this would query NewsAPI, Twitter, etc.
        
        # Base sentiment from returns
        df['sentiment_score'] = df['returns_1h'].rolling(window=12).mean() * 10
        df['sentiment_score'] = df['sentiment_score'].clip(-1, 1)
        
        # Add some noise and sector-specific patterns
        sector = SP500_SECTORS.get(symbol, "Unknown")
        
        if sector == "Technology":
            # Tech stocks more sensitive to sentiment
            df['sentiment_score'] *= 1.2
        elif sector == "Utilities":
            # Utilities less volatile
            df['sentiment_score'] *= 0.6
        elif sector == "Energy":
            # Energy follows oil sentiment
            df['sentiment_score'] += np.random.normal(0, 0.1, len(df))
            
        # Add sentiment volume (number of articles)
        df['sentiment_volume'] = np.random.poisson(5, len(df))
        
        # Sentiment consistency
        df['sentiment_consistency'] = df['sentiment_score'].rolling(window=20).std()
        
        return df
    
    def create_training_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create labels for training (future price movements)"""
        if df.empty:
            return df
            
        # Forward returns for different time horizons
        df['target_5m'] = df['close'].shift(-1) / df['close'] - 1
        df['target_15m'] = df['close'].shift(-3) / df['close'] - 1
        df['target_1h'] = df['close'].shift(-12) / df['close'] - 1
        
        # Classification labels
        df['direction_5m'] = (df['target_5m'] > 0).astype(int)
        df['direction_1h'] = (df['target_1h'] > 0).astype(int)
        
        # Volatility regime labels
        df['high_vol'] = (df['volatility_1h'] > df['volatility_1h'].quantile(0.75)).astype(int)
        
        # Contradiction labels (this is what PyG graphs will learn)
        df['contradiction_signal'] = 0  # Default: no contradiction
        
        # Overhype: positive sentiment but negative forward returns
        overhype_mask = (df['sentiment_score'] > 0.3) & (df['target_1h'] < -0.001)
        df.loc[overhype_mask, 'contradiction_signal'] = 1
        
        # Underhype: negative sentiment but positive forward returns
        underhype_mask = (df['sentiment_score'] < -0.3) & (df['target_1h'] > 0.001)
        df.loc[underhype_mask, 'contradiction_signal'] = 2
        
        return df
    
    def collect_all_data(self, num_workers: int = 4):
        """Collect data for all S&P 500 symbols in parallel"""
        logger.info(f"Starting parallel data collection with {num_workers} workers")
        
        all_data = []
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            future_to_symbol = {
                executor.submit(self.process_symbol, symbol): symbol 
                for symbol in self.symbols[:10]  # Start with first 10 for testing
            }
            
            # Process completed tasks with progress bar
            for future in tqdm(as_completed(future_to_symbol), 
                             total=len(future_to_symbol),
                             desc="Collecting S&P 500 data"):
                symbol = future_to_symbol[future]
                try:
                    df = future.result()
                    if not df.empty:
                        all_data.append(df)
                        self.save_symbol_data(symbol, df)
                except Exception as e:
                    logger.error(f"Failed to process {symbol}: {str(e)}")
        
        # Combine all data
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            logger.info(f"Collected {len(combined_df)} total records")
            
            # Save combined dataset
            self.save_combined_data(combined_df)
            
            return combined_df
        else:
            logger.error("No data collected!")
            return pd.DataFrame()
    
    def process_symbol(self, symbol: str) -> pd.DataFrame:
        """Process a single symbol through the full pipeline"""
        # Collect price data
        df = self.collect_price_data(symbol)
        if df.empty:
            return df
            
        # Calculate technical indicators
        df = self.calculate_technical_indicators(df)
        
        # Collect sentiment data
        df = self.collect_sentiment_data(symbol, df)
        
        # Create training labels
        df = self.create_training_labels(df)
        
        # Drop NaN rows from rolling calculations
        df = df.dropna()
        
        return df
    
    def save_symbol_data(self, symbol: str, df: pd.DataFrame):
        """Save individual symbol data"""
        symbol_path = DATA_DIR / "raw" / f"{symbol}.parquet"
        df.to_parquet(symbol_path, engine='pyarrow')
        logger.info(f"Saved {symbol} data to {symbol_path}")
    
    def save_combined_data(self, df: pd.DataFrame):
        """Save combined dataset for training"""
        # Save as parquet for efficient loading
        combined_path = DATA_DIR / "processed" / "sp500_training_data.parquet"
        df.to_parquet(combined_path, engine='pyarrow')
        logger.info(f"Saved combined data to {combined_path}")
        
        # Save metadata
        metadata = {
            "collection_date": datetime.now().isoformat(),
            "start_date": self.start_date,
            "end_date": self.end_date,
            "symbols": self.symbols,
            "total_records": len(df),
            "features": list(df.columns),
            "sectors": {sector: list(df[df['sector'] == sector]['symbol'].unique()) 
                       for sector in df['sector'].unique()}
        }
        
        metadata_path = DATA_DIR / "metadata" / "collection_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved metadata to {metadata_path}")
        
        # Print summary statistics
        logger.info("\n" + "="*50)
        logger.info("Data Collection Summary")
        logger.info("="*50)
        logger.info(f"Total records: {len(df):,}")
        logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
        logger.info(f"Symbols collected: {df['symbol'].nunique()}")
        logger.info("\nSector distribution:")
        for sector, count in df['sector'].value_counts().items():
            logger.info(f"  {sector}: {count:,} records")
        logger.info("\nContradiction signals:")
        for signal, count in df['contradiction_signal'].value_counts().items():
            signal_names = {0: "None", 1: "Overhype", 2: "Underhype"}
            logger.info(f"  {signal_names.get(signal, signal)}: {count:,} records")

def main():
    """Run the data collection pipeline"""
    print("S&P 500 Training Data Collection")
    print("="*50)
    
    # Check if we need to install dependencies
    try:
        import yfinance
    except ImportError:
        print("Installing required packages...")
        os.system("pip install yfinance pandas pyarrow tqdm")
        import yfinance
    
    # Initialize collector
    collector = SP500DataCollector(
        start_date="2022-01-01",  # 2 years of data
        end_date=None  # Up to today
    )
    
    # Collect all data
    df = collector.collect_all_data(num_workers=4)
    
    if not df.empty:
        print(f"\nSuccessfully collected {len(df):,} training samples!")
        print(f"Data saved to: {DATA_DIR}")
        print("\nNext steps:")
        print("1. Create training loop for Fusion Alpha")
        print("2. Build backtesting framework")
        print("3. Train on this historical data")
    else:
        print("Data collection failed!")

if __name__ == "__main__":
    main()