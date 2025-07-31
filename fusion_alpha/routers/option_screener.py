#!/usr/bin/env python3
"""
option_screener.py

Scans a list of tickers and, for each, uses predicted return and Sharpe (or estimated volatility)
to estimate an expected move and suggest a potential option strategy (call, put, or straddle).
This is a mock implementation.
Usage:
  python option_screener.py --tickers "AAPL,MSFT,GOOGL" --risk_free_rate 0.01 --target_mode normalized
"""
import argparse
import numpy as np
import pandas as pd
from options_utils import black_scholes_price

def parse_args():
    parser = argparse.ArgumentParser(description="Option screener based on predicted return and Sharpe.")
    parser.add_argument("--tickers", type=str, required=True, help="Comma-separated list of ticker symbols.")
    parser.add_argument("--risk_free_rate", type=float, default=0.0, help="Annualized risk-free rate.")
    parser.add_argument("--target_mode", type=str, default="normalized", choices=["normalized", "binary", "rolling"])
    return parser.parse_args()

def mock_predict(ticker):
    # For demonstration, simulate predicted return and estimated Sharpe.
    predicted_return = np.random.uniform(-0.02, 0.02)  # ±2%
    estimated_sharpe = np.random.uniform(0, 1)
    return predicted_return, estimated_sharpe

def screen_options(ticker, predicted_return, estimated_sharpe, S=100, T=0.0833, r=0.01):
    """
    Given predicted return and Sharpe, determine an option strategy.
    S: current price (default 100), T: time to expiration in years (default 1 month ~0.0833)
    r: risk-free rate.
    """
    # Estimate expected move as predicted_return * 100 (for percentage move).
    expected_move = abs(predicted_return)
    # For simplicity, if predicted return is positive (and strong), suggest a call.
    # If negative, suggest a put. If small, suggest a straddle.
    if abs(predicted_return) < 0.005:
        strategy = "Straddle"
    elif predicted_return > 0:
        strategy = "Call"
    else:
        strategy = "Put"
    
    # Use Black-Scholes to compute option price for an at-the-money option.
    K = S
    sigma = 0.2  # Use a constant implied volatility (or compute based on Sharpe if desired).
    option_price = black_scholes_price(S, K, T, r, sigma, option_type=strategy.lower() if strategy!="Straddle" else "call")
    return strategy, expected_move, option_price

def main():
    args = parse_args()
    tickers = [t.strip() for t in args.tickers.split(",")]
    opportunities = []
    for ticker in tickers:
        predicted_return, estimated_sharpe = mock_predict(ticker)
        strategy, expected_move, option_price = screen_options(ticker, predicted_return, estimated_sharpe)
        opp = {
            "ticker": ticker,
            "predicted_return": predicted_return,
            "estimated_sharpe": estimated_sharpe,
            "suggested_strategy": strategy,
            "expected_move": expected_move,
            "option_price": option_price
        }
        opportunities.append(opp)
        print(f"Ticker: {ticker} | Predicted Return: {predicted_return:.4f} | Sharpe: {estimated_sharpe:.4f} | Strategy: {strategy} | Option Price: {option_price:.2f}")
    
    df = pd.DataFrame(opportunities)
    output_csv = "option_opportunities.csv"
    df.to_csv(output_csv, index=False)
    print(f"Option opportunities saved to {output_csv}")

if __name__ == "__main__":
    main()