
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
from fusionnet import FusionNet
from contradiction_engine import ContradictionEngine
import matplotlib.pyplot as plt
import math

def parse_args():
    parser = argparse.ArgumentParser(description="Walk-forward validation for contradiction-aware trading.")
    parser.add_argument("--dataset_path", type=str, default="./training_data/dataset.npz", help="Path to dataset .npz file")
    parser.add_argument("--num_windows", type=int, default=5, help="Number of sequential windows")
    parser.add_argument("--target_mode", type=str, default="normalized", choices=["normalized", "binary", "rolling"])
    parser.add_argument("--epochs", type=int, default=50, help="Epochs per fold")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--threshold", type=float, default=0.01, help="Signal threshold")
    parser.add_argument("--initial_capital", type=float, default=1000.0, help="Starting capital")
    return parser.parse_args()

def simulate_trading(predictions, actual_returns, contradiction_tags, threshold, initial_capital):
    capital = initial_capital
    cumulative_log_return = 0.0
    equity_curve = [capital]
    trade_details = []
    daily_log_returns = []
    for i in range(len(predictions)):
        # Add noise
        noise = np.random.normal(0, 0.003)
        pred_noisy = predictions[i] + noise
        # False signal injection: 5% chance to reverse prediction
        if np.random.rand() < 0.05:
            pred_noisy = -pred_noisy
        action = "NO_TRADE"
        trade_executed = False
        
        if contradiction_tags[i] == "underhype" and pred_noisy > threshold:
            action = "LONG"
            trade_executed = True
            effective_return = actual_returns[i] - 0.002
            effective_return = np.clip(effective_return, -0.03, 0.03)
        elif contradiction_tags[i] == "overhype" and pred_noisy < -threshold:
            action = "SHORT"
            trade_executed = True
            effective_return = -actual_returns[i] - 0.002
            effective_return = np.clip(effective_return, -0.03, 0.03)
        else:
            effective_return = 0.0
        log_return = np.log(1 + effective_return)
        daily_log_returns.append(log_return)
        cumulative_log_return += log_return
        capital = initial_capital * np.exp(cumulative_log_return)
        equity_curve.append(capital)
        trade_details.append({
            "index": i,
            "raw_prediction": predictions[i],
            "noisy_prediction": pred_noisy,
            "actual_return": actual_returns[i],
            "contradiction_tag": contradiction_tags[i],
            "action": action,
            "effective_return": effective_return,
            "log_return": log_return,
            "cumulative_capital": capital
        })
    return np.array(equity_curve), trade_details, daily_log_returns

def compute_cagr(initial_capital, final_capital, num_days):
    years = num_days / 252.0
    return (final_capital / initial_capital) ** (1/years) - 1

def compute_max_drawdown(equity_curve):
    peaks = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - peaks) / peaks
    return np.min(drawdowns)

def compute_sharpe(daily_log_returns, risk_free_rate=0.0):
    arr = np.array(daily_log_returns)
    if arr.std() == 0:
        return 0.0
    return (arr.mean() - risk_free_rate) / arr.std() * np.sqrt(252)

def train_and_test(train_indices, test_indices, data, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Extract training data.
    tech_train = torch.tensor(data["technical_features"][train_indices], dtype=torch.float32).to(device)
    finbert_train = torch.tensor(data["finbert_embeddings"][train_indices], dtype=torch.float32).to(device)
    price_train = torch.tensor(data["price_movements"][train_indices], dtype=torch.float32).to(device)
    sentiment_train = torch.tensor(data["news_sentiment_scores"][train_indices], dtype=torch.float32).to(device)
    target_train = torch.tensor(data["target_returns"][train_indices], dtype=torch.float32).to(device)
    
    # Test data.
    tech_test = torch.tensor(data["technical_features"][test_indices], dtype=torch.float32).to(device)
    finbert_test = torch.tensor(data["finbert_embeddings"][test_indices], dtype=torch.float32).to(device)
    price_test = torch.tensor(data["price_movements"][test_indices], dtype=torch.float32).to(device)
    sentiment_test = torch.tensor(data["news_sentiment_scores"][test_indices], dtype=torch.float32).to(device)
    target_test = torch.tensor(data["target_returns"][test_indices], dtype=torch.float32).to(device)
    
    # Initialize model and contradiction engine.
    model = FusionNet(input_dim=tech_train.shape[1], hidden_dim=512, use_attention=True, fusion_method='concat', target_mode=args.target_mode).to(device)
    contr_engine = ContradictionEngine(embedding_dim=768).to(device)
    if args.target_mode == "binary":
        loss_fn = nn.BCEWithLogitsLoss()
    else:
        loss_fn = nn.MSELoss()
    optimizer = optim.Adam(list(model.parameters()) + list(contr_engine.parameters()), lr=1e-3)
    
    n_train = tech_train.shape[0]
    num_batches = (n_train + args.batch_size - 1) // args.batch_size
    
    for epoch in range(args.epochs):
        model.train()
        contr_engine.train()
        permutation = torch.randperm(n_train)
        epoch_loss = 0.0
        for i in range(num_batches):
            indices = permutation[i*args.batch_size : (i+1)*args.batch_size]
            batch_tech = tech_train[indices]
            batch_finbert = finbert_train[indices]
            batch_price = price_train[indices]
            batch_sentiment = sentiment_train[indices]
            batch_target = target_train[indices]
            
            optimizer.zero_grad()
            updated_embeddings = []
            for j in range(batch_finbert.size(0)):
                upd_emb, _ = contr_engine(batch_finbert[j], batch_tech[j], batch_price[j], batch_sentiment[j])
                updated_embeddings.append(upd_emb)
            updated_embeddings = torch.stack(updated_embeddings)
            preds = model(batch_tech, updated_embeddings).view(-1)
            loss = loss_fn(preds, batch_target.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item() * batch_finbert.size(0)
        avg_loss = epoch_loss / n_train
        print(f"Fold Training Epoch {epoch+1}/{args.epochs}: Loss = {avg_loss:.4f}")
    
    # Inference on test set.
    model.eval()
    contr_engine.eval()
    with torch.no_grad():
        updated_test_embeddings = []
        n_test = tech_test.shape[0]
        contradiction_tags = []
        for j in range(n_test):
            upd_emb, ctype = contr_engine(finbert_test[j], tech_test[j], price_test[j], sentiment_test[j])
            updated_test_embeddings.append(upd_emb)
            contradiction_tags.append(ctype if ctype is not None else "none")
        updated_test_embeddings = torch.stack(updated_test_embeddings)
        test_preds = model(tech_test, updated_test_embeddings).view(-1).cpu().numpy()
        test_targets = target_test.view(-1).cpu().numpy()
    return test_preds, test_targets, np.array(contradiction_tags)

def main():
    args = parse_args()
    # Load full dataset.
    data = np.load(args.dataset_path)
    N = data["technical_features"].shape[0]
    window_size = N // args.num_windows
    all_equity_curves = []
    all_metrics = []
    
    # Walk-forward validation: For each window, train on window i and test on window i+1.
    for i in range(args.num_windows - 1):
        train_start = i * window_size
        train_end = (i + 1) * window_size
        test_start = train_end
        test_end = (i + 2) * window_size if (i + 2) * window_size <= N else N
        
        print(f"Window {i+1}: Train [{train_start}:{train_end}], Test [{test_start}:{test_end}]")
        test_preds, test_targets, contradiction_tags = train_and_test(
            np.arange(train_start, train_end),
            np.arange(test_start, test_end),
            data, args)
        
        # Simulate trading on the test window.
        equity_curve, trade_details, daily_log_returns = simulate_trading(
            test_preds, test_targets, contradiction_tags, args.threshold, args.initial_capital)
        final_cap = equity_curve[-1]
        cagr = compute_cagr(args.initial_capital, final_cap, len(equity_curve)-1)
        sharpe = compute_sharpe(daily_log_returns)
        max_dd = compute_max_drawdown(equity_curve)
        
        metrics = {
            "final_capital": final_cap,
            "CAGR": cagr,
            "Sharpe": sharpe,
            "Max_Drawdown": max_dd,
            "trade_count": len(trade_details)
        }
        print(f"Window {i+1} Metrics: Final Cap = {final_cap:.2f}, CAGR = {cagr*100:.2f}%, Sharpe = {sharpe:.4f}, Max Drawdown = {max_dd*100:.2f}%")
        all_equity_curves.append(equity_curve)
        all_metrics.append(metrics)
    
    # Compute average metrics.
    final_caps = [m["final_capital"] for m in all_metrics]
    cagr_vals = [m["CAGR"] for m in all_metrics]
    sharpe_vals = [m["Sharpe"] for m in all_metrics]
    max_dd_vals = [m["Max_Drawdown"] for m in all_metrics]

    print(f"  Final Capital: {np.mean(final_caps):.2f}")
    print(f"  CAGR: {np.mean(cagr_vals)*100:.2f}%")
    print(f"  Sharpe Ratio: {np.mean(sharpe_vals):.4f}")
    print(f"  Max Drawdown: {np.mean(max_dd_vals)*100:.2f}%")
    
    # Plot average equity curve (for simplicity, plot equity curve from first window).
    plt.figure(figsize=(10,6))
    plt.plot(all_equity_curves[0], marker="o")
    plt.xlabel("Trading Days")
    plt.ylabel("Portfolio Value (log scale)")
    plt.yscale("log")
    plt.title("Equity Curve (First Window)")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
