#!/usr/bin/env python3
"""
validate_predictions_vs_close.py

Validates predicted returns against real market close-to-close returns.
Loads predictions, target_returns, and contradiction_tags from predictions_routed.npz,
and real OHLCV data from ./data/raw/<ticker>.csv. Computes:
  - Pearson correlation,
  - Directional accuracy,
  - Mean Absolute Error (MAE),
and prints 5 sample rows comparing predictions vs. actual close movement.
Usage:
  python validate_predictions_vs_close.py --ticker AAPL
"""
import argparse
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error

def parse_args():
    parser = argparse.ArgumentParser(description="Validate predictions against real OHLCV close data.")
    parser.add_argument("--npz_path", type=str, default="./training_data/predictions_routed.npz", help="Path to predictions_routed.npz")
    parser.add_argument("--ticker", type=str, required=True, help="Ticker symbol for OHLCV data (e.g., AAPL)")
    return parser.parse_args()

def main():
    args = parse_args()
    # Load predictions and targets.
    data = np.load(args.npz_path, allow_pickle=True)
    predictions = data["predictions"].flatten()
    target_returns = data["target_returns"].flatten()
    contradiction_tags = data["contradiction_tags"]
    
    # Load OHLCV data.
    ohlcv_path = f"./data/raw/{args.ticker}.csv"
    df = pd.read_csv(ohlcv_path, parse_dates=["Date"]).sort_values("Date")
    df = df.reset_index(drop=True)
    
    # Align predictions with OHLCV data.
    # Assume the dataset's order corresponds to trading days in the OHLCV CSV.
    # For each prediction, use the current day's close and next day's close.
    if len(predictions) >= len(df) - 1:
        predictions = predictions[:len(df)-1]
        target_returns = target_returns[:len(df)-1]
        contradiction_tags = contradiction_tags[:len(df)-1]
    else:
        print("Warning: Not enough predictions to cover all OHLCV days.")
    
    current_close = df["Close"].values[:-1]
    next_close = df["Close"].values[1:]
    actual_returns = (next_close - current_close) / current_close
    
    # Compute metrics.
    corr, _ = pearsonr(predictions, actual_returns)
    direction_accuracy = np.mean((predictions > 0) == (actual_returns > 0))
    mae = mean_absolute_error(actual_returns, predictions)
    
    print("Validation Metrics:")
    print(f"  Pearson Correlation: {corr:.4f}")
    print(f"  Directional Accuracy: {direction_accuracy*100:.2f}%")
    print(f"  Mean Absolute Error: {mae:.4f}")
    
    # Print 5 sample rows.
    sample_indices = np.linspace(0, len(predictions)-1, 5, dtype=int)
    print("\nSample Comparison (Prediction vs. Actual Close Return):")
    for idx in sample_indices:
        print(f"  Day {idx}: Prediction = {predictions[idx]:.4f}, Actual Return = {actual_returns[idx]:.4f}, Tag = {contradiction_tags[idx]}")
    
if __name__ == "__main__":
    main()