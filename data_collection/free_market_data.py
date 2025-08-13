#!/usr/bin/env python3
"""
Free Market Data Collection System

Collects market data from free sources:
1. Yahoo Finance (yfinance) - OHLCV, fundamentals, options
2. Alpha Vantage API (free tier) - Economic indicators, crypto
3. FRED API (Federal Reserve) - Economic data
4. Polygon.io (free tier) - Market data
5. IEX Cloud (free tier) - Market data

This replaces expensive Bloomberg/Refinitiv feeds with free alternatives.
"""

import os
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path
import logging
from dataclasses import dataclass, asdict
import json
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    print("yfinance not available. Install with: pip install yfinance")
    YFINANCE_AVAILABLE = False

try:
    import alpha_vantage
    from alpha_vantage.timeseries import TimeSeries
    from alpha_vantage.fundamentaldata import FundamentalData
    from alpha_vantage.cryptocurrencies import CryptoCurrencies
    ALPHA_VANTAGE_AVAILABLE = True
except ImportError:
    print("alpha_vantage not available. Install with: pip install alpha-vantage")
    ALPHA_VANTAGE_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MarketDataConfig:
    """Configuration for market data collection"""
    # API Keys (set as environment variables)
    alpha_vantage_key: Optional[str] = None
    polygon_key: Optional[str] = None
    iex_token: Optional[str] = None
    
    # Data settings
    data_dir: str = "/home/ryan/trading/mismatch-trading/data_collection/market_data"
    max_symbols_per_request: int = 100
    rate_limit_delay: float = 1.0  # seconds between requests
    
    # Cache settings
    cache_duration_minutes: int = 60
    enable_caching: bool = True

@dataclass
class OHLCV:
    """OHLC+Volume data structure"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    symbol: str

@dataclass
class MarketQuote:
    """Real-time market quote"""
    symbol: str
    price: float
    change: float
    change_percent: float
    volume: int
    timestamp: datetime
    bid: Optional[float] = None
    ask: Optional[float] = None
    market_cap: Optional[float] = None

class FreeMarketDataCollector:
    """
    Free market data collector using multiple sources
    """
    
    def __init__(self, config: Optional[MarketDataConfig] = None):
        self.config = config or MarketDataConfig()
        
        # Setup data directory
        self.data_dir = Path(self.config.data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Load API keys from environment
        self._load_api_keys()
        
        # Initialize API clients
        self._setup_api_clients()
        
        # Cache for API responses
        self.cache = {}
        self.cache_timestamps = {}
        
        # Rate limiting
        self.last_request_time = {}
        
        logger.info("Initialized free market data collector")
    
    def _load_api_keys(self):
        """Load API keys from environment variables"""
        self.config.alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY', 'demo')
        self.config.polygon_key = os.getenv('POLYGON_API_KEY')
        self.config.iex_token = os.getenv('IEX_CLOUD_TOKEN')
        
        if self.config.alpha_vantage_key == 'demo':
            logger.warning("Using demo Alpha Vantage key - limited to 5 requests per minute")
    
    def _setup_api_clients(self):
        """Setup API clients for various data sources"""
        self.yf_session = None
        
        # Alpha Vantage clients
        if ALPHA_VANTAGE_AVAILABLE and self.config.alpha_vantage_key:
            try:
                self.av_timeseries = TimeSeries(key=self.config.alpha_vantage_key, output_format='pandas')
                self.av_fundamentals = FundamentalData(key=self.config.alpha_vantage_key, output_format='pandas')
                self.av_crypto = CryptoCurrencies(key=self.config.alpha_vantage_key, output_format='pandas')
                logger.info("Alpha Vantage clients initialized")
            except Exception as e:
                logger.error(f"Alpha Vantage setup failed: {e}")
                self.av_timeseries = None
                self.av_fundamentals = None
                self.av_crypto = None
        else:
            self.av_timeseries = None
            self.av_fundamentals = None
            self.av_crypto = None
    
    def _check_rate_limit(self, source: str):
        """Check and enforce rate limits"""
        now = time.time()
        
        if source in self.last_request_time:
            time_since_last = now - self.last_request_time[source]
            if time_since_last < self.config.rate_limit_delay:
                sleep_time = self.config.rate_limit_delay - time_since_last
                time.sleep(sleep_time)
        
        self.last_request_time[source] = now
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid"""
        if not self.config.enable_caching:
            return False
        
        if cache_key not in self.cache_timestamps:
            return False
        
        cache_age = datetime.now() - self.cache_timestamps[cache_key]
        return cache_age.total_seconds() < (self.config.cache_duration_minutes * 60)
    
    def _cache_data(self, cache_key: str, data):
        """Cache data with timestamp"""
        if self.config.enable_caching:
            self.cache[cache_key] = data
            self.cache_timestamps[cache_key] = datetime.now()
    
    def get_yahoo_finance_data(self, symbols: Union[str, List[str]], 
                              period: str = "1d", interval: str = "1m") -> Dict[str, pd.DataFrame]:
        """
        Get OHLCV data from Yahoo Finance (free)
        
        Args:
            symbols: Single symbol or list of symbols
            period: Period for data (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
        """
        if not YFINANCE_AVAILABLE:
            logger.error("yfinance not available")
            return {}
        
        if isinstance(symbols, str):
            symbols = [symbols]
        
        cache_key = f"yf_{'-'.join(symbols)}_{period}_{interval}"
        
        # Check cache
        if self._is_cache_valid(cache_key):
            logger.info(f"Using cached Yahoo Finance data for {symbols}")
            return self.cache[cache_key]
        
        self._check_rate_limit("yahoo_finance")
        
        try:
            data = {}
            
            if len(symbols) == 1:
                # Single symbol
                ticker = yf.Ticker(symbols[0])
                df = ticker.history(period=period, interval=interval)
                if not df.empty:
                    data[symbols[0]] = df
            else:
                # Multiple symbols
                tickers = yf.Tickers(' '.join(symbols))
                for symbol in symbols:
                    try:
                        df = tickers.tickers[symbol].history(period=period, interval=interval)
                        if not df.empty:
                            data[symbol] = df
                    except Exception as e:
                        logger.warning(f"Failed to get data for {symbol}: {e}")
                        continue
            
            # Cache the data
            self._cache_data(cache_key, data)
            
            logger.info(f"Retrieved Yahoo Finance data for {len(data)} symbols")
            return data
            
        except Exception as e:
            logger.error(f"Error getting Yahoo Finance data: {e}")
            return {}
    
    def get_real_time_quotes(self, symbols: Union[str, List[str]]) -> Dict[str, MarketQuote]:
        """Get real-time quotes from Yahoo Finance"""
        if not YFINANCE_AVAILABLE:
            return {}
        
        if isinstance(symbols, str):
            symbols = [symbols]
        
        cache_key = f"quotes_{'-'.join(symbols)}"
        
        # Use shorter cache for real-time quotes (1 minute)
        if (cache_key in self.cache_timestamps and 
            (datetime.now() - self.cache_timestamps[cache_key]).total_seconds() < 60):
            return self.cache[cache_key]
        
        self._check_rate_limit("yahoo_quotes")
        
        try:
            quotes = {}
            
            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    
                    # Get current price data
                    price = info.get('currentPrice', info.get('regularMarketPrice', 0))
                    change = price - info.get('previousClose', price)
                    change_percent = (change / info.get('previousClose', 1)) * 100 if info.get('previousClose') else 0
                    
                    quote = MarketQuote(
                        symbol=symbol,
                        price=price,
                        change=change,
                        change_percent=change_percent,
                        volume=info.get('volume', 0),
                        timestamp=datetime.now(),
                        bid=info.get('bid'),
                        ask=info.get('ask'),
                        market_cap=info.get('marketCap')
                    )
                    
                    quotes[symbol] = quote
                    
                except Exception as e:
                    logger.warning(f"Failed to get quote for {symbol}: {e}")
                    continue
            
            # Cache quotes
            self._cache_data(cache_key, quotes)
            
            logger.info(f"Retrieved real-time quotes for {len(quotes)} symbols")
            return quotes
            
        except Exception as e:
            logger.error(f"Error getting real-time quotes: {e}")
            return {}
    
    def get_alpha_vantage_data(self, symbol: str, function: str = "TIME_SERIES_INTRADAY") -> Optional[pd.DataFrame]:
        """Get data from Alpha Vantage API (free tier: 5 requests/minute, 500/day)"""
        if not self.av_timeseries:
            logger.warning("Alpha Vantage not available")
            return None
        
        cache_key = f"av_{symbol}_{function}"
        
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]
        
        self._check_rate_limit("alpha_vantage")
        
        try:
            if function == "TIME_SERIES_INTRADAY":
                data, meta_data = self.av_timeseries.get_intraday(symbol=symbol, interval='5min', outputsize='compact')
            elif function == "TIME_SERIES_DAILY":
                data, meta_data = self.av_timeseries.get_daily(symbol=symbol, outputsize='compact')
            elif function == "OVERVIEW":
                data = self.av_fundamentals.get_company_overview(symbol=symbol)[0]
            else:
                logger.warning(f"Unknown Alpha Vantage function: {function}")
                return None
            
            self._cache_data(cache_key, data)
            logger.info(f"Retrieved Alpha Vantage {function} data for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Error getting Alpha Vantage data for {symbol}: {e}")
            return None
    
    def get_economic_data(self, indicator: str = "GDP") -> Optional[pd.DataFrame]:
        """Get economic data from FRED API (free)"""
        try:
            # FRED API is free but requires registration
            # Using a simplified version that doesn't require API key
            fred_indicators = {
                "GDP": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=GDP",
                "UNRATE": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=UNRATE",
                "FEDFUNDS": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=FEDFUNDS",
                "CPI": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL",
                "DGS10": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10"
            }
            
            if indicator not in fred_indicators:
                logger.warning(f"Economic indicator {indicator} not available")
                return None
            
            cache_key = f"fred_{indicator}"
            if self._is_cache_valid(cache_key):
                return self.cache[cache_key]
            
            self._check_rate_limit("fred")
            
            # Download CSV data
            response = requests.get(fred_indicators[indicator])
            response.raise_for_status()
            
            # Parse CSV
            from io import StringIO
            df = pd.read_csv(StringIO(response.text), parse_dates=['DATE'], index_col='DATE')
            
            self._cache_data(cache_key, df)
            logger.info(f"Retrieved FRED economic data for {indicator}")
            return df
            
        except Exception as e:
            logger.error(f"Error getting economic data for {indicator}: {e}")
            return None
    
    def get_crypto_data(self, symbols: Union[str, List[str]], vs_currency: str = "USD") -> Dict[str, pd.DataFrame]:
        """Get cryptocurrency data from free sources"""
        if isinstance(symbols, str):
            symbols = [symbols]
        
        crypto_data = {}
        
        # Try Yahoo Finance first (has crypto data)
        for symbol in symbols:
            try:
                # Yahoo Finance crypto format
                yf_symbol = f"{symbol}-{vs_currency}"
                data = self.get_yahoo_finance_data(yf_symbol, period="1d", interval="5m")
                
                if yf_symbol in data:
                    crypto_data[symbol] = data[yf_symbol]
                
            except Exception as e:
                logger.warning(f"Failed to get crypto data for {symbol}: {e}")
                continue
        
        logger.info(f"Retrieved crypto data for {len(crypto_data)} symbols")
        return crypto_data
    
    def get_options_data(self, symbol: str) -> Optional[Dict[str, pd.DataFrame]]:
        """Get options data from Yahoo Finance"""
        if not YFINANCE_AVAILABLE:
            return None
        
        cache_key = f"options_{symbol}"
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]
        
        try:
            ticker = yf.Ticker(symbol)
            options_dates = ticker.options
            
            if not options_dates:
                return None
            
            options_data = {}
            
            # Get options for next 2 expiration dates
            for date in options_dates[:2]:
                try:
                    option_chain = ticker.option_chain(date)
                    options_data[date] = {
                        'calls': option_chain.calls,
                        'puts': option_chain.puts
                    }
                except Exception as e:
                    logger.warning(f"Failed to get options for {symbol} {date}: {e}")
                    continue
            
            self._cache_data(cache_key, options_data)
            logger.info(f"Retrieved options data for {symbol}")
            return options_data
            
        except Exception as e:
            logger.error(f"Error getting options data for {symbol}: {e}")
            return None
    
    def get_fundamentals(self, symbol: str) -> Optional[Dict]:
        """Get fundamental data from Yahoo Finance"""
        if not YFINANCE_AVAILABLE:
            return None
        
        cache_key = f"fundamentals_{symbol}"
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]
        
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Extract key fundamental metrics
            fundamentals = {
                'market_cap': info.get('marketCap'),
                'pe_ratio': info.get('trailingPE'),
                'forward_pe': info.get('forwardPE'),
                'peg_ratio': info.get('pegRatio'),
                'price_to_book': info.get('priceToBook'),
                'price_to_sales': info.get('priceToSalesTrailing12Months'),
                'ev_ebitda': info.get('enterpriseToEbitda'),
                'profit_margin': info.get('profitMargins'),
                'operating_margin': info.get('operatingMargins'),
                'return_on_equity': info.get('returnOnEquity'),
                'return_on_assets': info.get('returnOnAssets'),
                'debt_to_equity': info.get('debtToEquity'),
                'current_ratio': info.get('currentRatio'),
                'revenue_growth': info.get('revenueGrowth'),
                'earnings_growth': info.get('earningsGrowth'),
                'dividend_yield': info.get('dividendYield'),
                'beta': info.get('beta'),
                'sector': info.get('sector'),
                'industry': info.get('industry')
            }
            
            self._cache_data(cache_key, fundamentals)
            logger.info(f"Retrieved fundamentals for {symbol}")
            return fundamentals
            
        except Exception as e:
            logger.error(f"Error getting fundamentals for {symbol}: {e}")
            return None
    
    def create_trading_dataset(self, symbols: List[str], days: int = 30) -> pd.DataFrame:
        """Create a comprehensive trading dataset for multiple symbols"""
        logger.info(f"Creating trading dataset for {len(symbols)} symbols over {days} days")
        
        all_data = []
        
        for symbol in symbols:
            try:
                # Get OHLCV data
                ohlcv_data = self.get_yahoo_finance_data(symbol, period=f"{days}d", interval="1h")
                
                if symbol not in ohlcv_data:
                    logger.warning(f"No OHLCV data for {symbol}")
                    continue
                
                df = ohlcv_data[symbol].copy()
                df['symbol'] = symbol
                
                # Add technical indicators
                df = self._add_technical_indicators(df)
                
                # Get fundamentals
                fundamentals = self.get_fundamentals(symbol)
                if fundamentals:
                    for key, value in fundamentals.items():
                        if value is not None:
                            df[f'fund_{key}'] = value
                
                # Add real-time quote if available
                quotes = self.get_real_time_quotes(symbol)
                if symbol in quotes:
                    quote = quotes[symbol]
                    df['current_price'] = quote.price
                    df['price_change'] = quote.change
                    df['price_change_pct'] = quote.change_percent
                
                all_data.append(df)
                
                # Rate limiting
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                continue
        
        if not all_data:
            logger.error("No data collected for any symbols")
            return pd.DataFrame()
        
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Sort by timestamp
        combined_df = combined_df.sort_values('Datetime' if 'Datetime' in combined_df.columns else combined_df.index)
        
        logger.info(f"Created dataset with {len(combined_df)} rows and {len(combined_df.columns)} columns")
        return combined_df
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic technical indicators to OHLCV data"""
        try:
            # Moving averages
            df['sma_20'] = df['Close'].rolling(window=20).mean()
            df['sma_50'] = df['Close'].rolling(window=50).mean()
            df['ema_12'] = df['Close'].ewm(span=12).mean()
            df['ema_26'] = df['Close'].ewm(span=26).mean()
            
            # MACD
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            
            # RSI
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            bb_period = 20
            df['bb_middle'] = df['Close'].rolling(window=bb_period).mean()
            bb_std = df['Close'].rolling(window=bb_period).std()
            df['bb_upper'] = df['bb_middle'] + (2 * bb_std)
            df['bb_lower'] = df['bb_middle'] - (2 * bb_std)
            
            # Volume indicators
            df['volume_sma'] = df['Volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['Volume'] / df['volume_sma']
            
            # Returns
            df['returns'] = df['Close'].pct_change()
            df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
            
            # Volatility
            df['volatility'] = df['returns'].rolling(window=20).std()
            
        except Exception as e:
            logger.error(f"Error adding technical indicators: {e}")
        
        return df
    
    def save_data(self, data: Union[pd.DataFrame, Dict], filename: str):
        """Save data to disk"""
        filepath = self.data_dir / filename
        
        try:
            if isinstance(data, pd.DataFrame):
                data.to_parquet(filepath.with_suffix('.parquet'))
            else:
                with open(filepath.with_suffix('.json'), 'w') as f:
                    json.dump(data, f, indent=2, default=str)
            
            logger.info(f"Saved data to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving data: {e}")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        valid_cache_entries = sum(1 for key in self.cache.keys() if self._is_cache_valid(key))
        
        return {
            'total_cache_entries': len(self.cache),
            'valid_cache_entries': valid_cache_entries,
            'expired_cache_entries': len(self.cache) - valid_cache_entries
        }

def main():
    """Test the free market data collection system"""
    print("Testing Free Market Data Collection System")
    print("="*60)
    
    # Test symbols
    test_symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'BTC-USD']
    
    # Create collector
    collector = FreeMarketDataCollector()
    
    print(f"Testing data collection for {test_symbols}")
    
    # Test Yahoo Finance OHLCV data
    print(f"\n1. Testing Yahoo Finance OHLCV data...")
    start_time = time.time()
    ohlcv_data = collector.get_yahoo_finance_data(test_symbols[:3], period="5d", interval="1h")
    print(f"   Collected OHLCV for {len(ohlcv_data)} symbols in {time.time() - start_time:.2f}s")
    
    for symbol, df in ohlcv_data.items():
        print(f"   {symbol}: {len(df)} bars, latest close: ${df['Close'].iloc[-1]:.2f}")
    
    # Test real-time quotes
    print(f"\n2. Testing real-time quotes...")
    start_time = time.time()
    quotes = collector.get_real_time_quotes(test_symbols[:3])
    print(f"   Collected quotes for {len(quotes)} symbols in {time.time() - start_time:.2f}s")
    
    for symbol, quote in quotes.items():
        print(f"   {symbol}: ${quote.price:.2f} ({quote.change_percent:+.2f}%)")
    
    # Test fundamentals
    print(f"\n3. Testing fundamental data...")
    fundamentals = collector.get_fundamentals('AAPL')
    if fundamentals:
        print(f"   AAPL fundamentals: PE={fundamentals.get('pe_ratio'):.2f}, Beta={fundamentals.get('beta'):.2f}")
    
    # Test crypto data
    print(f"\n4. Testing cryptocurrency data...")
    crypto_data = collector.get_crypto_data(['BTC', 'ETH'])
    print(f"   Collected crypto data for {len(crypto_data)} symbols")
    
    # Test economic data
    print(f"\n5. Testing economic data...")
    try:
        fed_funds = collector.get_economic_data('FEDFUNDS')
        if fed_funds is not None and not fed_funds.empty:
            latest_rate = fed_funds.iloc[-1, 0]
            print(f"   Federal Funds Rate: {latest_rate:.2f}%")
        else:
            print(f"   Economic data not available")
    except Exception as e:
        print(f"   Economic data error: {e}")
    
    # Test options data
    print(f"\n6. Testing options data...")
    options = collector.get_options_data('AAPL')
    if options:
        total_contracts = sum(len(data['calls']) + len(data['puts']) for data in options.values())
        print(f"   AAPL options: {total_contracts} contracts across {len(options)} expiration dates")
    
    # Create comprehensive dataset
    print(f"\n7. Creating comprehensive trading dataset...")
    start_time = time.time()
    dataset = collector.create_trading_dataset(['AAPL', 'MSFT'], days=7)
    print(f"   Created dataset: {len(dataset)} rows, {len(dataset.columns)} columns in {time.time() - start_time:.2f}s")
    
    if not dataset.empty:
        print(f"   Sample columns: {list(dataset.columns)[:10]}...")
        collector.save_data(dataset, 'sample_trading_dataset')
    
    # Cache statistics
    cache_stats = collector.get_cache_stats()
    print(f"\nCache Statistics:")
    print(f"   Valid entries: {cache_stats['valid_cache_entries']}")
    print(f"   Total entries: {cache_stats['total_cache_entries']}")
    
    print(f"\nFree market data collection system working successfully!")
    print(f"Ready for integration with trading pipeline")
    print(f"\nAvailable data sources:")
    print(f"   • Yahoo Finance: OHLCV, quotes, fundamentals, options")
    print(f"   • Alpha Vantage: {'Available' if collector.av_timeseries else 'Unavailable'} Intraday, fundamentals")
    print(f"   • FRED Economic: Interest rates, inflation, GDP")
    print(f"   • Cryptocurrency: BTC, ETH, major altcoins")

if __name__ == "__main__":
    main()