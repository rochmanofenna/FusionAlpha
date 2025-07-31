#!/usr/bin/env python
"""
FusionAlpha Quick Start Example
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pandas as pd
from datetime import datetime, timedelta

# Import components
from fusion_alpha.pipelines.contradiction_engine import ContradictionEngine
from fusion_alpha.models.finbert import RealFinBERT
from data_collection.free_market_data import get_stock_data
from data_collection.free_news_collector import collect_news

def main():
    """Basic system workflow"""
    
    print("FusionAlpha Quick Start")
    print("=" * 40)
    
    # Setup
    print("\n1. Initializing system...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Device: {device}")
    
    # Models
    engine = ContradictionEngine()
    sentiment_model = RealFinBERT()
    
    # Market data
    print("\n2. Fetching market data...")
    symbol = "AAPL"
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    try:
        market_data = get_stock_data(symbol, start_date, end_date)
        latest_price = market_data['Close'].iloc[-1]
        price_change = (market_data['Close'].iloc[-1] - market_data['Close'].iloc[-2]) / market_data['Close'].iloc[-2]
        print(f"   {symbol} price: ${latest_price:.2f}")
        print(f"   Change: {price_change:.2%}")
    except Exception as e:
        print(f"   Error: {e}")
        price_change = -0.02
        print("   Using demo data")
    
    # News analysis
    print("\n3. Processing news...")
    try:
        news_data = collect_news(symbol, max_articles=5)
        if news_data:
            latest_news = news_data[0]['title'] + " " + news_data[0].get('description', '')
            sentiment_output = sentiment_model.get_sentiment(latest_news)
            sentiment_score = sentiment_output['positive'] - sentiment_output['negative']
            print(f"   Sentiment: {sentiment_score:.2f}")
            print(f"   News: {news_data[0]['title'][:60]}...")
        else:
            sentiment_score = 0.8
            print("   Using demo sentiment")
    except Exception as e:
        print(f"   Error: {e}")
        sentiment_score = 0.8
        print("   Using demo sentiment")
    
    # Signal processing
    print("\n4. Generating signals...")
    
    dummy_embedding = torch.randn(768)
    dummy_technical = torch.randn(10)
    
    updated_embedding, signal_type = engine(
        dummy_embedding,
        dummy_technical,
        torch.tensor(price_change),
        torch.tensor(sentiment_score)
    )
    
    # Output
    print("\n5. Trading Signal:")
    print("=" * 40)
    
    if signal_type:
        print(f"SIGNAL DETECTED: {signal_type.upper()}")
        
        if signal_type == "overhype":
            print("Action: SHORT")
            print("Reason: Signal divergence detected")
        elif signal_type == "underhype":
            print("Action: LONG")
            print("Reason: Signal convergence detected")
    else:
        print("No signal detected")
        print("Market conditions normal")
    
    print("\n" + "=" * 40)
    print("Demo complete. Review results before trading.")

if __name__ == "__main__":
    main()