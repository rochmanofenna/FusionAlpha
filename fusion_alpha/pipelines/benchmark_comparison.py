import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from scipy.stats import pearsonr

def simulate_trading(predictions, target_returns, contradiction_tags, threshold, initial_capital):
    capital = initial_capital
    cumulative_log_return = 0.0
    equity_curve = [capital]
    daily_log_returns = []
    for i in range(len(predictions)):
        noise = np.random.normal(0, 0.003)
        pred_noisy = predictions[i] + noise
        if np.random.rand() < 0.05:
            pred_noisy = -pred_noisy
        action = "NO_TRADE"
        if contradiction_tags[i] == "underhype" and pred_noisy > threshold:
            effective_return = target_returns[i] - 0.002
            effective_return = np.clip(effective_return, -0.03, 0.03)
        elif contradiction_tags[i] == "overhype" and pred_noisy < -threshold:
            effective_return = -target_returns[i] - 0.002
            effective_return = np.clip(effective_return, -0.03, 0.03)
        else:
            effective_return = 0.0
        log_return = np.log(1 + effective_return)
        cumulative_log_return += log_return
        daily_log_returns.append(log_return)
        capital = initial_capital * np.exp(cumulative_log_return)
        equity_curve.append(capital)
    return equity_curve, daily_log_returns

def compute_cagr(initial, final, num_days):
    years = num_days / 252.0
    return (final / initial) ** (1/years) - 1

def compute_max_drawdown(equity_curve):
    equity_curve = np.array(equity_curve)
    peaks = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - peaks) / peaks
    return np.min(drawdowns)

def compute_sharpe(daily_log_returns, risk_free_rate):
    arr = np.array(daily_log_returns)
    if arr.std() == 0:
        return 0.0
    adjusted_rf = risk_free_rate / 252.0
    return (arr.mean() - adjusted_rf) / (arr.std() + 1e-8) * np.sqrt(252)

def simple_sma_strategy(ohlcv, initial_capital):
    df = ohlcv.copy()
    df["SMA10"] = df["Close"].rolling(window=10, min_periods=1).mean()
    df["SMA30"] = df["Close"].rolling(window=30, min_periods=1).mean()
    signals = df["SMA10"] > df["SMA30"]
    df["Return"] = df["Close"].pct_change().fillna(0)
    strategy_returns = df["Return"].where(signals, -df["Return"])
    equity_curve, daily_log_returns = simulate_trading(strategy_returns.values, strategy_returns.values, np.array(["none"]*len(strategy_returns)), 0, initial_capital)
    return equity_curve, daily_log_returns

def always_long_strategy(returns, initial_capital):
    return simulate_trading(returns, returns, np.array(["none"]*len(returns)), 0, initial_capital)

def main():
    parser = argparse.ArgumentParser(description="Benchmark comparison of trading strategies.")
    parser.add_argument("--dataset_path", type=str, default="./training_data/dataset.npz", help="Path to dataset.npz")
    parser.add_argument("--initial_capital", type=float, default=1000.0)
    parser.add_argument("--threshold", type=float, default=0.01, help="Signal threshold (not used in baseline strategies)")
    parser.add_argument("--risk_free_rate", type=float, default=0.0, help="Annualized risk-free rate")
    args = parser.parse_args()
    
    # Load dataset.
    data = np.load(args.dataset_path)
    target_returns = data["target_returns"].flatten()  # actual returns
    # For benchmark strategies, simulate OHLCV data.
    num_days = len(target_returns)
    dates = pd.date_range(start="2020-01-01", periods=num_days, freq="B")
    close_prices = 100 + np.cumsum(target_returns) * 100  # simplistic simulation.
    ohlcv = pd.DataFrame({
        "Date": dates,
        "Open": close_prices * (1 + np.random.uniform(-0.005, 0.005, num_days)),
        "High": close_prices * (1 + np.random.uniform(0, 0.01, num_days)),
        "Low": close_prices * (1 - np.random.uniform(0, 0.01, num_days)),
        "Close": close_prices,
        "Volume": np.random.randint(1000000, 5000000, num_days)
    }).set_index("Date")
    
    # Backtest benchmark strategies.
    eq_always_long, lr_always = always_long_strategy(ohlcv["Close"].pct_change().fillna(0).values, args.initial_capital)
    eq_sma, lr_sma = simple_sma_strategy(ohlcv, args.initial_capital)
    
    # For contradiction-aware predictions, load predictions_routed.npz.
    preds_data = np.load("./training_data/predictions_routed.npz", allow_pickle=True)
    preds = preds_data["predictions"]
    tags = preds_data["contradiction_tags"]
    # Filter out "none" samples.
    mask = (np.array(tags) != "none")
    preds_filtered = preds[mask]
    actual_filtered = data["target_returns"].flatten()[mask]
    eq_contra, lr_contra = simulate_trading(preds_filtered, actual_filtered, np.array(tags)[mask], args.threshold, args.initial_capital)    
    
    def metrics(eq_curve, daily_lr):
        final = eq_curve[-1]
        cagr = compute_cagr(args.initial_capital, final, len(eq_curve)-1)
        sharpe = compute_sharpe(daily_lr, args.risk_free_rate)
        max_dd = compute_max_drawdown(eq_curve)
        return final, cagr, sharpe, max_dd
    
    final_contra, cagr_contra, sharpe_contra, max_dd_contra = metrics(eq_contra, lr_contra)
    final_always, cagr_always, sharpe_always, max_dd_always = metrics(eq_always_long, lr_always)
    final_sma, cagr_sma, sharpe_sma, max_dd_sma = metrics(eq_sma, lr_sma)
    
    print("Benchmark Results:")
    print("Contradiction-Aware Strategy:")
    print(f"  Final Portfolio: ${final_contra:.2f}, CAGR: {cagr_contra*100:.2f}%, Sharpe: {sharpe_contra:.4f}, Max Drawdown: {max_dd_contra*100:.2f}%")
    print("Always Long Strategy:")
    print(f"  Final Portfolio: ${final_always:.2f}, CAGR: {cagr_always*100:.2f}%, Sharpe: {sharpe_always:.4f}, Max Drawdown: {max_dd_always*100:.2f}%")
    print("SMA Crossover Strategy:")
    print(f"  Final Portfolio: ${final_sma:.2f}, CAGR: {cagr_sma*100:.2f}%, Sharpe: {sharpe_sma:.4f}, Max Drawdown: {max_dd_sma*100:.2f}%")
    
    # --- HFT vs. Swing Feasibility Note ---
    risk_free_rate = args.risk_free_rate
    average_holding_time = 1  # day (assumed)
    # Simulate an intraday volatility measure.
    intraday_volatility = np.random.uniform(0.01, 0.03)
    # Here, using the contradiction-aware strategy's Sharpe as an example.
    print("\nHFT vs. Swing Feasibility Note:")
    print(f"  Average Holding Time: {average_holding_time} day(s)")
    print(f"  Strategy Sharpe Ratio (adjusted): {sharpe_contra:.4f} vs. Intraday Volatility: {intraday_volatility:.4f}")
    print("  Suggestion: This strategy's returns are better suited for low-volatility swing trading rather than high-frequency trading.")
    # --- End HFT vs. Swing Note ---
    
    # Plot equity curves.
    plt.figure(figsize=(10,6))
    plt.plot(eq_contra, label="Contradiction-Aware")
    plt.plot(eq_always_long, label="Always Long")
    plt.plot(eq_sma, label="SMA Crossover")
    plt.xlabel("Trading Days")
    plt.ylabel("Portfolio Value")
    plt.title("Benchmark Equity Curves")
    plt.legend()
    plt.grid(True)
    plt.yscale("log")
    plt.savefig("./training_data/benchmark_equity_curve.png")
    plt.show()
    print("Equity curve plot saved to ./training_data/benchmark_equity_curve.png")
    
    # Save metrics to .npz.
    metrics_dict = {
        "contradiction_final": final_contra,
        "contradiction_cagr": cagr_contra,
        "contradiction_sharpe": sharpe_contra,
        "contradiction_max_dd": max_dd_contra,
        "always_long_final": final_always,
        "always_long_cagr": cagr_always,
        "always_long_sharpe": sharpe_always,
        "always_long_max_dd": max_dd_always,
        "sma_final": final_sma,
        "sma_cagr": cagr_sma,
        "sma_sharpe": sharpe_sma,
        "sma_max_dd": max_dd_sma
    }
    np.savez("./training_data/benchmark_metrics.npz", **metrics_dict)
    print("Benchmark metrics saved to ./training_data/benchmark_metrics.npz")

if __name__ == "__main__":
    main()