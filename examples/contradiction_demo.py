#!/usr/bin/env python
"""
FusionAlpha Contradiction Detection Demo

This script demonstrates the core theory behind FusionAlpha:
detecting contradictions between news sentiment and price movements.
"""

import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Simulate the contradiction detection process
class ContradictionDemo:
    def __init__(self):
        self.sentiment_threshold_pos = 0.7
        self.sentiment_threshold_neg = -0.7
        self.price_threshold = 0.02  # 2% movement
        
    def generate_sample_data(self, n_days=30):
        """Generate synthetic market data for demonstration"""
        dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')
        
        # Generate synthetic price data
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, n_days)
        prices = 100 * np.exp(np.cumsum(returns))
        
        # Generate synthetic sentiment scores
        sentiments = np.random.uniform(-1, 1, n_days)
        
        # Inject some contradictions
        contradiction_indices = [5, 12, 18, 25]
        for idx in contradiction_indices:
            if idx < n_days:
                # Create overhype: positive sentiment, negative price movement
                if idx % 2 == 0:
                    sentiments[idx] = np.random.uniform(0.7, 0.9)
                    returns[idx] = np.random.uniform(-0.03, -0.02)
                # Create underhype: negative sentiment, positive price movement
                else:
                    sentiments[idx] = np.random.uniform(-0.9, -0.7)
                    returns[idx] = np.random.uniform(0.02, 0.03)
        
        # Recalculate prices with injected contradictions
        prices = 100 * np.exp(np.cumsum(returns))
        
        return pd.DataFrame({
            'date': dates,
            'price': prices,
            'returns': returns,
            'sentiment': sentiments
        })
    
    def detect_contradictions(self, df):
        """Detect contradictions in the data"""
        contradictions = []
        
        for idx, row in df.iterrows():
            sentiment = row['sentiment']
            price_movement = row['returns']
            
            # Overhype detection
            if sentiment > self.sentiment_threshold_pos and price_movement < -self.price_threshold:
                contradictions.append({
                    'date': row['date'],
                    'type': 'overhype',
                    'sentiment': sentiment,
                    'price_movement': price_movement,
                    'signal': 'SELL'  # Market overreacted to positive news
                })
            
            # Underhype detection
            elif sentiment < self.sentiment_threshold_neg and price_movement > self.price_threshold:
                contradictions.append({
                    'date': row['date'],
                    'type': 'underhype',
                    'sentiment': sentiment,
                    'price_movement': price_movement,
                    'signal': 'BUY'  # Market underreacted to negative news
                })
        
        return pd.DataFrame(contradictions)
    
    def visualize_results(self, df, contradictions_df):
        """Create visualization of the contradiction detection"""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        
        # Price chart
        ax1.plot(df['date'], df['price'], 'b-', label='Price')
        ax1.set_ylabel('Price ($)')
        ax1.set_title('FusionAlpha Contradiction Detection Demo')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Sentiment chart
        ax2.plot(df['date'], df['sentiment'], 'g-', label='Sentiment Score')
        ax2.axhline(y=self.sentiment_threshold_pos, color='r', linestyle='--', alpha=0.5)
        ax2.axhline(y=self.sentiment_threshold_neg, color='r', linestyle='--', alpha=0.5)
        ax2.set_ylabel('Sentiment')
        ax2.set_ylim(-1.1, 1.1)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Returns with contradictions highlighted
        ax3.bar(df['date'], df['returns'], color='gray', alpha=0.5, label='Daily Returns')
        
        # Highlight contradictions
        for _, contradiction in contradictions_df.iterrows():
            date = contradiction['date']
            idx = df[df['date'] == date].index[0]
            
            if contradiction['type'] == 'overhype':
                ax3.bar(date, df.loc[idx, 'returns'], color='red', label='Overhype' if idx == 0 else '')
                ax3.annotate('SELL', xy=(date, df.loc[idx, 'returns']), 
                           xytext=(0, -20), textcoords='offset points',
                           ha='center', fontsize=8, color='red')
            else:
                ax3.bar(date, df.loc[idx, 'returns'], color='green', label='Underhype' if idx == 0 else '')
                ax3.annotate('BUY', xy=(date, df.loc[idx, 'returns']), 
                           xytext=(0, 20), textcoords='offset points',
                           ha='center', fontsize=8, color='green')
        
        ax3.set_ylabel('Returns (%)')
        ax3.set_xlabel('Date')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def calculate_strategy_performance(self, df, contradictions_df):
        """Calculate performance metrics for the contradiction-based strategy"""
        # Simple strategy: Buy on underhype, Sell on overhype
        df['position'] = 0
        
        for _, contradiction in contradictions_df.iterrows():
            date_idx = df[df['date'] == contradiction['date']].index[0]
            if contradiction['signal'] == 'BUY':
                df.loc[date_idx:, 'position'] = 1
            else:  # SELL signal
                df.loc[date_idx:, 'position'] = 0
        
        # Calculate strategy returns
        df['strategy_returns'] = df['position'].shift(1) * df['returns']
        df['cumulative_returns'] = (1 + df['returns']).cumprod()
        df['cumulative_strategy_returns'] = (1 + df['strategy_returns']).cumprod()
        
        # Performance metrics
        total_return = df['cumulative_strategy_returns'].iloc[-1] - 1
        sharpe_ratio = np.sqrt(252) * df['strategy_returns'].mean() / df['strategy_returns'].std()
        max_drawdown = (df['cumulative_strategy_returns'] / df['cumulative_strategy_returns'].cummax() - 1).min()
        win_rate = (df[df['strategy_returns'] > 0]['strategy_returns'].count() / 
                   df[df['strategy_returns'] != 0]['strategy_returns'].count())
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'num_trades': len(contradictions_df)
        }

def main():
    print("=" * 60)
    print("FusionAlpha Contradiction Detection Demo")
    print("=" * 60)
    
    # Initialize demo
    demo = ContradictionDemo()
    
    # Generate sample data
    print("\n1. Generating synthetic market data...")
    df = demo.generate_sample_data(n_days=60)
    print(f"   Generated {len(df)} days of data")
    
    # Detect contradictions
    print("\n2. Detecting contradictions...")
    contradictions_df = demo.detect_contradictions(df)
    print(f"   Found {len(contradictions_df)} contradictions:")
    for _, c in contradictions_df.iterrows():
        print(f"   - {c['date'].strftime('%Y-%m-%d')}: {c['type'].upper()} "
              f"(Sentiment: {c['sentiment']:.2f}, Price Move: {c['price_movement']:.2%})")
    
    # Calculate performance
    print("\n3. Calculating strategy performance...")
    performance = demo.calculate_strategy_performance(df, contradictions_df)
    
    print("\n4. Performance Metrics:")
    print(f"   - Total Return: {performance['total_return']:.2%}")
    print(f"   - Sharpe Ratio: {performance['sharpe_ratio']:.2f}")
    print(f"   - Max Drawdown: {performance['max_drawdown']:.2%}")
    print(f"   - Win Rate: {performance['win_rate']:.2%}")
    print(f"   - Number of Trades: {performance['num_trades']}")
    
    # Visualize results
    print("\n5. Creating visualization...")
    fig = demo.visualize_results(df, contradictions_df)
    plt.savefig('examples/contradiction_demo_results.png', dpi=300, bbox_inches='tight')
    print("   Saved visualization to examples/contradiction_demo_results.png")
    
    # Show key insight
    print("\n" + "=" * 60)
    print("KEY INSIGHT:")
    print("The FusionAlpha system identifies moments when market sentiment")
    print("and price action diverge, creating trading opportunities.")
    print("- OVERHYPE: Positive news but price drops → SELL signal")
    print("- UNDERHYPE: Negative news but price rises → BUY signal")
    print("=" * 60)

if __name__ == "__main__":
    main()