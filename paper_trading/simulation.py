#!/usr/bin/env python3
"""
Paper Trading Simulation System

A complete paper trading system for testing the Fusion Alpha strategy in real-time:
1. Real-time data feed simulation
2. Order execution simulation with realistic fills
3. Portfolio tracking and risk management
4. Performance monitoring and alerts
5. Trade logging and audit trails
6. Integration with live data sources
"""

import os
import sys
import time
import threading
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import logging
from dataclasses import dataclass, asdict
import json
from enum import Enum
import queue

# Add project paths
sys.path.append('/home/ryan/trading/mismatch-trading')

try:
    from data_collection.free_market_data import FreeMarketDataCollector
    from fusion_alpha.models.real_finbert import get_finbert_processor
    from fusion_alpha.pipelines.end_to_end_integration import IntegratedTradingPipeline
    COMPONENTS_AVAILABLE = True
except ImportError:
    print("⚠️ Some components not available, using mock implementations")
    COMPONENTS_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

@dataclass
class Order:
    """Trading order"""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    timestamp: datetime = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    commission: float = 0.0
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class Position:
    """Portfolio position"""
    symbol: str
    quantity: float
    avg_price: float
    current_price: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    
    def update_current_price(self, price: float):
        self.current_price = price
        self.unrealized_pnl = (price - self.avg_price) * self.quantity

@dataclass
class Portfolio:
    """Portfolio state"""
    cash: float
    positions: Dict[str, Position]
    total_value: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    
    def update_total_value(self):
        position_value = sum(pos.quantity * pos.current_price for pos in self.positions.values())
        self.total_value = self.cash + position_value
        self.unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())

@dataclass
class PaperTradingConfig:
    """Configuration for paper trading"""
    # Initial settings
    initial_cash: float = 100000.0
    
    # Trading universe
    symbols: List[str] = None
    
    # Risk management
    max_position_size: float = 0.1  # 10% of portfolio
    max_portfolio_leverage: float = 1.0
    stop_loss_pct: float = 0.05  # 5%
    take_profit_pct: float = 0.15  # 15%
    
    # Execution settings
    commission_rate: float = 0.001  # 0.1%
    bid_ask_spread: float = 0.001   # 0.1%
    slippage_rate: float = 0.0005   # 0.05%
    
    # Data settings
    data_frequency: str = "1m"  # 1m, 5m, 1h
    lookback_window: int = 100
    
    # Output settings
    output_dir: str = "/home/ryan/trading/mismatch-trading/paper_trading/results"
    log_trades: bool = True
    real_time_alerts: bool = True

class MarketDataSimulator:
    """Simulates real-time market data feed"""
    
    def __init__(self, symbols: List[str], data_frequency: str = "1m"):
        self.symbols = symbols
        self.data_frequency = data_frequency
        self.subscribers = []
        self.running = False
        self.data_thread = None
        
        # Initialize data collector
        if COMPONENTS_AVAILABLE:
            self.data_collector = FreeMarketDataCollector()
        else:
            self.data_collector = None
        
        # Current prices
        self.current_prices = {}
        
        logger.info(f"Market data simulator initialized for {len(symbols)} symbols")
    
    def subscribe(self, callback):
        """Subscribe to market data updates"""
        self.subscribers.append(callback)
    
    def start(self):
        """Start the market data feed"""
        self.running = True
        self.data_thread = threading.Thread(target=self._data_loop, daemon=True)
        self.data_thread.start()
        logger.info("Market data feed started")
    
    def stop(self):
        """Stop the market data feed"""
        self.running = False
        if self.data_thread:
            self.data_thread.join()
        logger.info("Market data feed stopped")
    
    def _data_loop(self):
        """Main data feed loop"""
        while self.running:
            try:
                # Get current market data
                if self.data_collector:
                    quotes = self.data_collector.get_real_time_quotes(self.symbols)
                    
                    for symbol, quote in quotes.items():
                        self.current_prices[symbol] = {
                            'price': quote.price,
                            'bid': quote.bid or quote.price * 0.9995,
                            'ask': quote.ask or quote.price * 1.0005,
                            'volume': quote.volume,
                            'timestamp': quote.timestamp
                        }
                else:
                    # Generate mock data
                    for symbol in self.symbols:
                        if symbol not in self.current_prices:
                            self.current_prices[symbol] = {'price': 100.0}
                        
                        # Random walk
                        current = self.current_prices[symbol]['price']
                        change = np.random.normal(0, 0.002) * current
                        new_price = max(1.0, current + change)
                        
                        self.current_prices[symbol] = {
                            'price': new_price,
                            'bid': new_price * 0.9995,
                            'ask': new_price * 1.0005,
                            'volume': np.random.randint(1000, 10000),
                            'timestamp': datetime.now()
                        }
                
                # Notify subscribers
                for callback in self.subscribers:
                    try:
                        callback(self.current_prices)
                    except Exception as e:
                        logger.error(f"Error in market data callback: {e}")
                
                # Sleep based on frequency
                if self.data_frequency == "1m":
                    time.sleep(60)
                elif self.data_frequency == "5m":
                    time.sleep(300)
                else:
                    time.sleep(10)  # 10 seconds for testing
                
            except Exception as e:
                logger.error(f"Error in market data loop: {e}")
                time.sleep(5)
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol"""
        return self.current_prices.get(symbol, {}).get('price')

class OrderExecutionEngine:
    """Simulates realistic order execution"""
    
    def __init__(self, config: PaperTradingConfig, market_data: MarketDataSimulator):
        self.config = config
        self.market_data = market_data
        self.pending_orders = []
        self.order_history = []
        self.order_counter = 0
        
    def submit_order(self, symbol: str, side: OrderSide, order_type: OrderType, 
                    quantity: float, price: Optional[float] = None,
                    stop_price: Optional[float] = None) -> Order:
        """Submit a new order"""
        self.order_counter += 1
        order_id = f"ORDER_{self.order_counter:06d}"
        
        order = Order(
            order_id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price
        )
        
        # Validate order
        if not self._validate_order(order):
            order.status = OrderStatus.REJECTED
            logger.warning(f"Order rejected: {order}")
            return order
        
        # Add to pending orders
        self.pending_orders.append(order)
        logger.info(f"Order submitted: {order.order_id} {order.side.value} {order.quantity} {order.symbol}")
        
        return order
    
    def _validate_order(self, order: Order) -> bool:
        """Validate order parameters"""
        if order.quantity <= 0:
            return False
        
        if order.order_type == OrderType.LIMIT and order.price is None:
            return False
        
        if order.order_type in [OrderType.STOP, OrderType.STOP_LIMIT] and order.stop_price is None:
            return False
        
        return True
    
    def process_orders(self, market_data: Dict[str, Dict]) -> List[Order]:
        """Process pending orders against market data"""
        filled_orders = []
        
        for order in self.pending_orders[:]:  # Copy list to avoid modification during iteration
            if self._try_fill_order(order, market_data):
                filled_orders.append(order)
                self.pending_orders.remove(order)
                self.order_history.append(order)
        
        return filled_orders
    
    def _try_fill_order(self, order: Order, market_data: Dict[str, Dict]) -> bool:
        """Try to fill a pending order"""
        if order.symbol not in market_data:
            return False
        
        symbol_data = market_data[order.symbol]
        current_price = symbol_data['price']
        bid_price = symbol_data.get('bid', current_price * 0.9995)
        ask_price = symbol_data.get('ask', current_price * 1.0005)
        
        fill_price = None
        
        if order.order_type == OrderType.MARKET:
            # Market orders fill immediately at current bid/ask
            if order.side == OrderSide.BUY:
                fill_price = ask_price
            else:
                fill_price = bid_price
        
        elif order.order_type == OrderType.LIMIT:
            # Limit orders fill if price is reached
            if order.side == OrderSide.BUY and ask_price <= order.price:
                fill_price = min(order.price, ask_price)
            elif order.side == OrderSide.SELL and bid_price >= order.price:
                fill_price = max(order.price, bid_price)
        
        # Add more order types as needed...
        
        if fill_price is not None:
            # Apply slippage
            if order.side == OrderSide.BUY:
                fill_price *= (1 + self.config.slippage_rate)
            else:
                fill_price *= (1 - self.config.slippage_rate)
            
            # Calculate commission
            trade_value = order.quantity * fill_price
            commission = trade_value * self.config.commission_rate
            
            # Fill the order
            order.status = OrderStatus.FILLED
            order.filled_quantity = order.quantity
            order.avg_fill_price = fill_price
            order.commission = commission
            
            logger.info(f"Order filled: {order.order_id} at ${fill_price:.4f}, commission: ${commission:.2f}")
            return True
        
        return False
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order"""
        for order in self.pending_orders:
            if order.order_id == order_id:
                order.status = OrderStatus.CANCELLED
                self.pending_orders.remove(order)
                self.order_history.append(order)
                logger.info(f"Order cancelled: {order_id}")
                return True
        return False

class PaperTradingSimulator:
    """Main paper trading simulator"""
    
    def __init__(self, config: PaperTradingConfig):
        self.config = config
        
        # Setup output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize portfolio
        self.portfolio = Portfolio(
            cash=config.initial_cash,
            positions={}
        )
        
        # Initialize components
        self.symbols = config.symbols or ['AAPL', 'MSFT', 'GOOGL', 'TSLA']
        self.market_data = MarketDataSimulator(self.symbols, config.data_frequency)
        self.execution_engine = OrderExecutionEngine(config, self.market_data)
        
        # Initialize trading pipeline
        if COMPONENTS_AVAILABLE:
            try:
                self.trading_pipeline = IntegratedTradingPipeline()
                logger.info("Trading pipeline initialized")
            except Exception as e:
                logger.warning(f"Trading pipeline initialization failed: {e}")
                self.trading_pipeline = None
        else:
            self.trading_pipeline = None
        
        # Performance tracking
        self.performance_history = []
        self.trade_log = []
        
        # Subscribe to market data
        self.market_data.subscribe(self._on_market_data)
        
        # Control flags
        self.running = False
        
        logger.info(f"Paper trading simulator initialized with ${config.initial_cash:,.2f}")
    
    def _on_market_data(self, market_data: Dict[str, Dict]):
        """Handle market data updates"""
        try:
            # Update portfolio positions
            for symbol, data in market_data.items():
                if symbol in self.portfolio.positions:
                    self.portfolio.positions[symbol].update_current_price(data['price'])
            
            # Update total portfolio value
            self.portfolio.update_total_value()
            
            # Process pending orders
            filled_orders = self.execution_engine.process_orders(market_data)
            
            # Update positions based on filled orders
            for order in filled_orders:
                self._update_position_from_order(order)
                
                # Log trade
                if self.config.log_trades:
                    self._log_trade(order)
            
            # Generate trading signals and execute trades
            if self.trading_pipeline and self.running:
                self._execute_trading_logic(market_data)
            
            # Record performance
            self._record_performance()
            
            # Check risk limits
            self._check_risk_limits()
            
        except Exception as e:
            logger.error(f"Error processing market data: {e}")
    
    def _update_position_from_order(self, order: Order):
        """Update portfolio position from filled order"""
        symbol = order.symbol
        
        if symbol not in self.portfolio.positions:
            self.portfolio.positions[symbol] = Position(
                symbol=symbol,
                quantity=0.0,
                avg_price=0.0,
                current_price=order.avg_fill_price
            )
        
        position = self.portfolio.positions[symbol]
        
        if order.side == OrderSide.BUY:
            # Calculate new average price
            total_cost = position.quantity * position.avg_price + order.filled_quantity * order.avg_fill_price
            position.quantity += order.filled_quantity
            position.avg_price = total_cost / position.quantity if position.quantity > 0 else 0
            
            # Update cash
            self.portfolio.cash -= order.filled_quantity * order.avg_fill_price + order.commission
            
        else:  # SELL
            # Update realized P&L
            realized_pnl = (order.avg_fill_price - position.avg_price) * order.filled_quantity - order.commission
            self.portfolio.realized_pnl += realized_pnl
            position.realized_pnl += realized_pnl
            
            # Update position
            position.quantity -= order.filled_quantity
            
            # Update cash
            self.portfolio.cash += order.filled_quantity * order.avg_fill_price - order.commission
            
            # Remove position if quantity is zero
            if abs(position.quantity) < 1e-6:
                del self.portfolio.positions[symbol]
    
    def _execute_trading_logic(self, market_data: Dict[str, Dict]):
        """Execute trading logic using the integrated pipeline"""
        try:
            # Prepare data for pipeline
            symbols_to_process = list(market_data.keys())
            batch_size = len(symbols_to_process)
            
            if batch_size == 0:
                return
            
            # Create mock inputs (in production, these would be real features)
            import torch
            
            finbert_embeddings = torch.randn(batch_size, 768)
            tech_features = torch.randn(batch_size, 10)
            price_movements = torch.tensor([
                (market_data[symbol]['price'] - 100) / 100  # Normalized price movement
                for symbol in symbols_to_process
            ])
            sentiment_scores = torch.randn(batch_size) * 0.5  # Mock sentiment
            
            # Get predictions from pipeline
            results = self.trading_pipeline.forward(
                finbert_embeddings, tech_features, price_movements, sentiment_scores
            )
            
            predictions = results['predictions']
            
            # Generate trades based on predictions
            for i, symbol in enumerate(symbols_to_process):
                prediction = predictions[i].item()
                current_price = market_data[symbol]['price']
                
                # Simple trading logic
                position_size = abs(prediction) * self.config.max_position_size * self.portfolio.total_value
                
                if abs(prediction) > 0.01:  # Minimum threshold
                    if prediction > 0:  # Buy signal
                        quantity = position_size / current_price
                        self.execution_engine.submit_order(
                            symbol, OrderSide.BUY, OrderType.MARKET, quantity
                        )
                    else:  # Sell signal
                        # Check if we have position to sell
                        if symbol in self.portfolio.positions:
                            current_quantity = self.portfolio.positions[symbol].quantity
                            sell_quantity = min(current_quantity, position_size / current_price)
                            if sell_quantity > 0:
                                self.execution_engine.submit_order(
                                    symbol, OrderSide.SELL, OrderType.MARKET, sell_quantity
                                )
            
        except Exception as e:
            logger.error(f"Error in trading logic: {e}")
    
    def _log_trade(self, order: Order):
        """Log completed trade"""
        trade_record = {
            'timestamp': order.timestamp.isoformat(),
            'order_id': order.order_id,
            'symbol': order.symbol,
            'side': order.side.value,
            'quantity': order.filled_quantity,
            'price': order.avg_fill_price,
            'commission': order.commission,
            'portfolio_value': self.portfolio.total_value
        }
        
        self.trade_log.append(trade_record)
        
        # Save to file
        if self.config.log_trades:
            trade_log_path = self.output_dir / 'trade_log.jsonl'
            with open(trade_log_path, 'a') as f:
                f.write(json.dumps(trade_record) + '\n')
    
    def _record_performance(self):
        """Record current performance metrics"""
        performance_record = {
            'timestamp': datetime.now().isoformat(),
            'total_value': self.portfolio.total_value,
            'cash': self.portfolio.cash,
            'unrealized_pnl': self.portfolio.unrealized_pnl,
            'realized_pnl': self.portfolio.realized_pnl,
            'total_pnl': self.portfolio.unrealized_pnl + self.portfolio.realized_pnl,
            'return_pct': (self.portfolio.total_value - self.config.initial_cash) / self.config.initial_cash,
            'num_positions': len(self.portfolio.positions)
        }
        
        self.performance_history.append(performance_record)
    
    def _check_risk_limits(self):
        """Check and enforce risk limits"""
        # Check individual position sizes
        for symbol, position in self.portfolio.positions.items():
            position_value = position.quantity * position.current_price
            position_pct = abs(position_value) / self.portfolio.total_value
            
            if position_pct > self.config.max_position_size:
                logger.warning(f"Position size limit exceeded for {symbol}: {position_pct:.2%}")
                # Could implement automatic position reduction here
        
        # Check stop losses
        for symbol, position in self.portfolio.positions.items():
            if position.quantity > 0:  # Long position
                pnl_pct = (position.current_price - position.avg_price) / position.avg_price
                if pnl_pct < -self.config.stop_loss_pct:
                    logger.warning(f"Stop loss triggered for {symbol}: {pnl_pct:.2%}")
                    # Submit stop loss order
                    self.execution_engine.submit_order(
                        symbol, OrderSide.SELL, OrderType.MARKET, position.quantity
                    )
    
    def start_trading(self):
        """Start the paper trading simulation"""
        logger.info("Starting paper trading simulation...")
        self.running = True
        self.market_data.start()
        
        # Run for a specified duration or until stopped
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Simulation interrupted by user")
        finally:
            self.stop_trading()
    
    def stop_trading(self):
        """Stop the paper trading simulation"""
        logger.info("Stopping paper trading simulation...")
        self.running = False
        self.market_data.stop()
        
        # Save final results
        self.save_results()
    
    def save_results(self):
        """Save trading results"""
        logger.info("Saving paper trading results...")
        
        # Save performance history
        performance_df = pd.DataFrame(self.performance_history)
        performance_df.to_csv(self.output_dir / 'performance_history.csv', index=False)
        
        # Save trade log
        if self.trade_log:
            trade_df = pd.DataFrame(self.trade_log)
            trade_df.to_csv(self.output_dir / 'trades.csv', index=False)
        
        # Save final portfolio state
        portfolio_summary = {
            'final_value': self.portfolio.total_value,
            'initial_value': self.config.initial_cash,
            'total_return': (self.portfolio.total_value - self.config.initial_cash) / self.config.initial_cash,
            'realized_pnl': self.portfolio.realized_pnl,
            'unrealized_pnl': self.portfolio.unrealized_pnl,
            'num_trades': len(self.trade_log),
            'positions': {symbol: asdict(pos) for symbol, pos in self.portfolio.positions.items()}
        }
        
        with open(self.output_dir / 'portfolio_summary.json', 'w') as f:
            json.dump(portfolio_summary, f, indent=2)
        
        logger.info(f"Results saved to {self.output_dir}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get current performance summary"""
        if not self.performance_history:
            return {}
        
        current_value = self.portfolio.total_value
        total_return = (current_value - self.config.initial_cash) / self.config.initial_cash
        
        # Calculate daily returns if we have enough data
        if len(self.performance_history) > 1:
            values = [p['total_value'] for p in self.performance_history]
            returns = [(values[i] - values[i-1]) / values[i-1] for i in range(1, len(values))]
            
            if returns:
                volatility = np.std(returns) * np.sqrt(252)  # Annualized
                sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
            else:
                volatility = 0
                sharpe_ratio = 0
        else:
            volatility = 0
            sharpe_ratio = 0
        
        return {
            'current_value': current_value,
            'total_return': total_return,
            'realized_pnl': self.portfolio.realized_pnl,
            'unrealized_pnl': self.portfolio.unrealized_pnl,
            'num_trades': len(self.trade_log),
            'num_positions': len(self.portfolio.positions),
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'cash': self.portfolio.cash
        }
    
    def print_status(self):
        """Print current trading status"""
        summary = self.get_performance_summary()
        
        print(f"\n{'='*50}")
        print(f"PAPER TRADING STATUS")
        print(f"{'='*50}")
        print(f"Portfolio Value: ${summary.get('current_value', 0):,.2f}")
        print(f"Total Return:    {summary.get('total_return', 0):.2%}")
        print(f"Realized P&L:    ${summary.get('realized_pnl', 0):,.2f}")
        print(f"Unrealized P&L:  ${summary.get('unrealized_pnl', 0):,.2f}")
        print(f"Cash:            ${summary.get('cash', 0):,.2f}")
        print(f"Positions:       {summary.get('num_positions', 0)}")
        print(f"Trades:          {summary.get('num_trades', 0)}")
        print(f"Sharpe Ratio:    {summary.get('sharpe_ratio', 0):.2f}")
        print(f"{'='*50}")

def main():
    """Run paper trading simulation"""
    print("🔧 Testing Paper Trading Simulation")
    print("="*50)
    
    # Create configuration
    config = PaperTradingConfig(
        initial_cash=50000.0,
        symbols=['AAPL', 'MSFT', 'GOOGL'],
        data_frequency="10s",  # Fast for testing
        max_position_size=0.2,
        log_trades=True
    )
    
    # Create simulator
    simulator = PaperTradingSimulator(config)
    
    # Print initial status
    simulator.print_status()
    
    try:
        print(f"\n🚀 Starting paper trading simulation...")
        print(f"   Initial cash: ${config.initial_cash:,.2f}")
        print(f"   Symbols: {config.symbols}")
        print(f"   Press Ctrl+C to stop")
        
        # Run for a short time for testing
        simulator.running = True
        simulator.market_data.start()
        
        # Run for 30 seconds for testing
        for i in range(30):
            time.sleep(1)
            if i % 10 == 0:
                simulator.print_status()
        
        # Stop and show results
        simulator.stop_trading()
        simulator.print_status()
        
        print(f"\n✅ Paper trading simulation completed!")
        print(f"📊 Trades executed: {len(simulator.trade_log)}")
        print(f"💾 Results saved to: {simulator.output_dir}")
        
    except KeyboardInterrupt:
        print(f"\n⏹️ Simulation stopped by user")
        simulator.stop_trading()
        simulator.print_status()
    
    except Exception as e:
        print(f"❌ Simulation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()