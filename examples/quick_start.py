#!/usr/bin/env python
"""
FusionAlpha Quick Start Example

This script shows how to use the FusionAlpha system for
real-time contradiction detection and trading signals.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pandas as pd
from datetime import datetime, timedelta

# Import FusionAlpha components
from fusion_alpha.pipelines.contradiction_engine import ContradictionEngine
from fusion_alpha.models.finbert import RealFinBERT
from data_collection.free_market_data import get_stock_data
from data_collection.free_news_collector import collect_news

def main():
    """Quick start example showing the basic workflow"""
    
    print("FusionAlpha Quick Start")
    print("=" * 50)
    
    # 1. Setup
    print("\n1. Setting up models...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Using device: {device}")
    
    # Initialize models
    contradiction_engine = ContradictionEngine()
    finbert = RealFinBERT()
    
    # 2. Get market data
    print("\n2. Fetching market data...")
    symbol = "AAPL"
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    try:
        market_data = get_stock_data(symbol, start_date, end_date)
        latest_price = market_data['Close'].iloc[-1]
        price_change = (market_data['Close'].iloc[-1] - market_data['Close'].iloc[-2]) / market_data['Close'].iloc[-2]
        print(f"   {symbol} latest price: ${latest_price:.2f}")
        print(f"   24h change: {price_change:.2%}")
    except Exception as e:
        print(f"   Error fetching market data: {e}")
        # Use dummy data for demo
        price_change = -0.02
        print("   Using dummy data for demonstration")
    
    # 3. Get news sentiment
    print("\n3. Analyzing news sentiment...")
    try:
        news_data = collect_news(symbol, max_articles=5)
        if news_data:
            # Get sentiment for latest news
            latest_news = news_data[0]['title'] + " " + news_data[0].get('description', '')
            sentiment_output = finbert.get_sentiment(latest_news)
            sentiment_score = sentiment_output['positive'] - sentiment_output['negative']
            print(f"   Latest news sentiment: {sentiment_score:.2f}")
            print(f"   News: {news_data[0]['title'][:80]}...")
        else:
            # Use dummy sentiment for demo
            sentiment_score = 0.8
            print("   Using dummy sentiment for demonstration")
    except Exception as e:
        print(f"   Error analyzing sentiment: {e}")
        sentiment_score = 0.8
        print("   Using dummy sentiment for demonstration")
    
    # 4. Detect contradictions
    print("\n4. Checking for contradictions...")
    
    # Create dummy tensors for the demo
    dummy_embedding = torch.randn(768)
    dummy_technical = torch.randn(10)
    
    # Run contradiction detection
    updated_embedding, contradiction_type = contradiction_engine(
        dummy_embedding,
        dummy_technical,
        torch.tensor(price_change),
        torch.tensor(sentiment_score)
    )
    
    # 5. Generate trading signal
    print("\n5. Trading Signal:")
    print("=" * 50)
    
    if contradiction_type:
        print(f"⚠️  CONTRADICTION DETECTED: {contradiction_type.upper()}")
        
        if contradiction_type == "overhype":
            print("📉 Signal: SELL or SHORT")
            print("   Reason: Positive sentiment but price is falling")
            print("   Action: Market may be overreacting to positive news")
        elif contradiction_type == "underhype":
            print("📈 Signal: BUY or LONG")
            print("   Reason: Negative sentiment but price is rising")
            print("   Action: Market may be undervaluing despite bad news")
    else:
        print("✅ No contradiction detected")
        print("   Market sentiment aligns with price movement")
    
    print("\n" + "=" * 50)
    print("Note: This is a demonstration. Always do your own research")
    print("before making trading decisions.")

if __name__ == "__main__":
    main()