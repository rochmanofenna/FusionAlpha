
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Backtest contradiction-aware trading strategy.")
    parser.add_argument("--npz_path", type=str, default="./training_data/predictions_routed.npz", help="Path to routed predictions .npz file.")
    parser.add_argument("--contradiction_filter", type=str, default="both", choices=["underhype", "overhype", "both"], help="Which contradiction types to include in trading.")
    parser.add_argument("--threshold", type=float, default=0.01, help="Minimum absolute predicted return to trigger trade.")
    parser.add_argument("--initial_capital", type=float, default=1000.0, help="Starting portfolio capital.")
    parser.add_argument("--output_csv", type=str, default="trades_real.csv", help="CSV file to save trade details.")
    return parser.parse_args()

def load_data(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    predictions = data["predictions"]       # predicted returns (1D)
    target_returns = data["target_returns"].flatten()  # actual returns (1D)
    contradiction_tags = data["contradiction_tags"]
    return predictions, target_returns, contradiction_tags

def filter_data(predictions, target_returns, contradiction_tags, filter_type):
    # Only include "underhype" and "overhype" if filter_type is "both"
    tags = np.array(contradiction_tags)
    if filter_type == "both":
        mask = (tags == "underhype") | (tags == "overhype")
    else:
        mask = (tags == filter_type)
    filtered_predictions = predictions[mask]
    filtered_targets = target_returns[mask]
    filtered_tags = tags[mask]
    return filtered_predictions, filtered_targets, filtered_tags

def simulate_trading(predictions, target_returns, contradiction_tags, threshold, initial_capital):
    capital = initial_capital
    cumulative_log_return = 0.0
    equity_curve = [capital]
    trade_details = []
    trade_counts = {"underhype": {"count": 0, "wins": 0}, "overhype": {"count": 0, "wins": 0}}
    daily_log_returns = []
    
    N = len(predictions)
    
    for i in range(N):
        # Add prediction noise.
        noise = np.random.normal(0, 0.003)
        pred_noisy = predictions[i] + noise
        
        # False signal injection: 5% chance to reverse sign.
        if np.random.rand() < 0.05:
            pred_noisy = -pred_noisy
        
        # Determine trade action.
        action = "NO_TRADE"
        trade_executed = False
        actual_return = target_returns[i]
        
        # For long trade: tag == "underhype" and pred_noisy > threshold.
        if contradiction_tags[i] == "underhype" and pred_noisy > threshold:
            action = "LONG"
            trade_counts["underhype"]["count"] += 1
            # Adjust actual return: subtract slippage.
            effective_return = actual_return - 0.002
            # Clip to ±3%.
            effective_return = np.clip(effective_return, -0.03, 0.03)
            trade_executed = True
            # Determine win: if effective_return > 0.
            if effective_return > 0:
                trade_counts["underhype"]["wins"] += 1
        # For short trade: tag == "overhype" and pred_noisy < -threshold.
        elif contradiction_tags[i] == "overhype" and pred_noisy < -threshold:
            action = "SHORT"
            trade_counts["overhype"]["count"] += 1
            # For short trades, effective return = -actual_return - slippage.
            effective_return = -actual_return - 0.002
            effective_return = np.clip(effective_return, -0.03, 0.03)
            trade_executed = True
            if effective_return > 0:
                trade_counts["overhype"]["wins"] += 1
        else:
            effective_return = 0.0
        
        # Compute log return.
        log_return = np.log(1 + effective_return)
        daily_log_returns.append(log_return)
        cumulative_log_return += log_return
        capital = initial_capital * np.exp(cumulative_log_return)
        equity_curve.append(capital)
        
        trade_details.append({
            "index": i,
            "predicted_return_raw": predictions[i],
            "predicted_return_noisy": pred_noisy,
            "target_return": actual_return,
            "contradiction_tag": contradiction_tags[i],
            "trade_action": action,
            "effective_return": effective_return,
            "log_return": log_return,
            "cumulative_capital": capital
        })
    
    return np.array(equity_curve), trade_details, trade_counts, daily_log_returns

def compute_cagr(initial_capital, final_capital, num_days):
    years = num_days / 252.0  # Assume 252 trading days per year.
    return (final_capital / initial_capital) ** (1/years) - 1

def compute_max_drawdown(equity_curve):
    equity_curve = np.array(equity_curve)
    peaks = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - peaks) / peaks
    return np.min(drawdowns)

# Example update for Sharpe ratio:
def compute_sharpe(daily_log_returns, risk_free_rate):
    arr = np.array(daily_log_returns)
    if arr.std() == 0:
        return 0.0
    adjusted_rf = risk_free_rate / 252.0
    return (arr.mean() - adjusted_rf) / (arr.std() + 1e-8) * np.sqrt(252)

def save_trade_details(trade_details, output_csv):
    df = pd.DataFrame(trade_details)
    df.to_csv(output_csv, index=False)
    print("Trade details saved to", output_csv)

def save_metrics(equity_curve, final_capital, cagr, sharpe, max_drawdown, trade_counts, output_path):
    np.savez(output_path,
             equity_curve=equity_curve,
             final_capital=final_capital,
             cagr=cagr,
             sharpe=sharpe,
             max_drawdown=max_drawdown,
             trade_counts=trade_counts)
    print("Backtest metrics saved to", output_path)

def plot_equity_curve(equity_curve):
    plt.figure(figsize=(10, 6))
    plt.plot(equity_curve, marker="o")
    plt.xlabel("Trading Days")
    plt.ylabel("Portfolio Value (log scale)")
    plt.yscale("log")
    plt.title("Equity Curve (Log Scale)")
    plt.grid(True)
    plt.show()

def main():
    args = parse_args()
    
    # Load routed predictions.
    predictions, target_returns, contradiction_tags = load_data(args.npz_path)
    
    # Filter: only include "underhype" and "overhype" or specific type.
    tags = np.array(contradiction_tags)
    if args.contradiction_filter == "both":
        mask = (tags == "underhype") | (tags == "overhype")
    else:
        mask = tags == args.contradiction_filter
    predictions = predictions[mask]
    target_returns = target_returns[mask]
    contradiction_tags = tags[mask]
    print(f"After filtering, {len(predictions)} samples remain.")
    
    # Simulate trades with realistic adjustments.
    equity_curve, trade_details, trade_counts, daily_log_returns = simulate_trading(
        predictions, target_returns, contradiction_tags, args.threshold, args.initial_capital)
    
    final_capital = equity_curve[-1]
    num_days = len(equity_curve) - 1
    cagr = compute_cagr(args.initial_capital, final_capital, num_days)
    sharpe = compute_sharpe(daily_log_returns)
    max_dd = compute_max_drawdown(equity_curve)
    
    print("Backtest Results:")
    print(f"  Final Portfolio Value: ${final_capital:.2f}")
    print(f"  CAGR: {cagr*100:.2f}%")
    print(f"  Sharpe Ratio: {sharpe:.4f}")
    print(f"  Max Drawdown: {max_dd*100:.2f}%")
    print("Trade Counts:", trade_counts)
    for t in trade_counts:
        if trade_counts[t]["count"] > 0:
            win_rate = trade_counts[t]["wins"] / trade_counts[t]["count"]
            print(f"  {t.capitalize()} Win Rate: {win_rate*100:.2f}%")
    
    # Save trade details CSV.
    save_trade_details(trade_details, args.output_csv)
    # Save overall backtest metrics.
    save_metrics(equity_curve, final_capital, cagr, sharpe, max_dd, trade_counts, "./training_data/backtest_metrics.npz")
    # Plot equity curve.
    plot_equity_curve(equity_curve)

if __name__ == "__main__":
    main()
