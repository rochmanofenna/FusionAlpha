
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
    main()import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveContradictionEngine(nn.Module):
    def __init__(self, embedding_dim=768):
        """
        Initializes learnable thresholds for contradiction detection.
        """
        super(AdaptiveContradictionEngine, self).__init__()
        self.embedding_dim = embedding_dim

        # Initialize thresholds as trainable parameters.
        self.pos_sent_thresh = nn.Parameter(torch.tensor(0.1))
        self.neg_sent_thresh = nn.Parameter(torch.tensor(-0.1))
        self.drop_thresh = nn.Parameter(torch.tensor(0.01))
        self.rise_thresh = nn.Parameter(torch.tensor(0.01))
        
        # A learned nonlinear transformation to generate updated semantic embeddings.
        self.transform = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.Tanh()
        )

    def forward(self, finbert_embedding, technical_features, price_movement, news_sentiment_score):
        # Convert sentiment and movement to scalars if needed.
        sentiment = news_sentiment_score.item() if isinstance(news_sentiment_score, torch.Tensor) and news_sentiment_score.dim() == 0 else news_sentiment_score
        movement = price_movement.item() if isinstance(price_movement, torch.Tensor) and price_movement.dim() == 0 else price_movement
        
        # Debug: print inputs and thresholds.
        print("ContradictionEngine Debug:")
        print(f"  Sentiment: {sentiment}, Price Movement: {movement}")
        print(f"  Thresholds -> pos: {self.pos_sent_thresh.item()}, neg: {self.neg_sent_thresh.item()}, drop: {self.drop_thresh.item()}, rise: {self.rise_thresh.item()}")
        
        contradiction_detected = False
        contradiction_type = None

        if sentiment > self.pos_sent_thresh and movement < -self.drop_thresh:
            contradiction_detected = True
            contradiction_type = "overhype"
        elif sentiment < self.neg_sent_thresh and movement > self.rise_thresh:
            contradiction_detected = True
            contradiction_type = "underhype"
        
        if contradiction_detected:
            print(f"  Contradiction detected: {contradiction_type}")
            updated_embedding = self.transform(finbert_embedding)
            return updated_embedding, contradiction_type
        else:
            print("  No contradiction detected.")
            return finbert_embedding, None

# Alias for backward compatibility
ContradictionEngine = AdaptiveContradictionEngine

if __name__ == "__main__":
    engine = ContradictionEngine()
    dummy_embedding = torch.randn(768)
    dummy_tech = torch.randn(10)
    price_movement = torch.tensor(-0.05)
    news_sentiment_score = torch.tensor(0.8)
    
    updated_emb, ctype = engine(dummy_embedding, dummy_tech, price_movement, news_sentiment_score)
    print("Contradiction type:", ctype)
    print("First 5 elements of updated embedding:", updated_emb[:5])import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# Component 1: EncoderTechnical
# ---------------------------
class EncoderTechnical(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_prob=0.1):
        """
        Encodes OHLCV + technical indicators (e.g., SMA, EMA, RSI, MACD, etc.).
        """
        super(EncoderTechnical, self).__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob) if dropout_prob > 0 else None
        
    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        if self.dropout:
            x = self.dropout(x)
        return x

# ---------------------------
# Component 2: EncoderSentiment
# ---------------------------
class EncoderSentiment(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_prob=0.1):
        """
        Encodes sentiment embeddings (e.g., from FinBERT or similar).
        """
        super(EncoderSentiment, self).__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob) if dropout_prob > 0 else None
        
    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        if self.dropout:
            x = self.dropout(x)
        return x

# ---------------------------
# Component 3: ProjectionHead
# ---------------------------
class ProjectionHead(nn.Module):
    def __init__(self, input_dim, proj_dim):
        """
        Projects features from an encoder into a shared latent space.
        """
        super(ProjectionHead, self).__init__()
        self.fc = nn.Linear(input_dim, proj_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        return x

# ---------------------------
# Component 4: AdaptiveFusion
# ---------------------------
class AdaptiveFusion(nn.Module):
    def __init__(self, latent_dim):
        """
        Fuses the projected embeddings using a gating mechanism.
        The gate network uses the cosine similarity-based contradiction score.
        """
        super(AdaptiveFusion, self).__init__()
        self.gate = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()  # Produces a weight between 0 and 1.
        )
        
    def forward(self, emb1, emb2):
        # Compute cosine similarity between the two embeddings.
        cos_sim = F.cosine_similarity(emb1, emb2, dim=1, eps=1e-8)  # Shape: (batch,)
        contradiction_score = 1.0 - cos_sim  # Higher value indicates more divergence.
        
        # Process contradiction score through the gating network.
        gate_input = contradiction_score.unsqueeze(1)  # Shape: (batch, 1)
        gate_weight = self.gate(gate_input)  # Weight for emb1; shape: (batch, 1)
        
        # Fuse the embeddings using the gate weight:
        # fused = weight * emb1 + (1 - weight) * emb2.
        fused = gate_weight * emb1 + (1 - gate_weight) * emb2
        return fused, contradiction_score, gate_weight

# ---------------------------
# Component 5: DecisionHead (FusionNet)
# ---------------------------
class DecisionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1):
        """
        Takes the fused latent representation and outputs a scalar prediction.
        """
        super(DecisionHead, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# ---------------------------
# Component 6: ContradictionLoss Module
# ---------------------------
class ContradictionLoss(nn.Module):
    def __init__(self, weight=1.0):
        """
        Penalizes high-confidence predictions when the projected embeddings diverge.
        """
        super(ContradictionLoss, self).__init__()
        self.weight = weight
        
    def forward(self, emb1, emb2, prediction):
        # Compute cosine similarity and derive contradiction score.
        cos_sim = F.cosine_similarity(emb1, emb2, dim=1, eps=1e-8)
        contradiction_score = 1.0 - cos_sim  # Shape: (batch,)
        # Use the absolute value of the prediction as a proxy for confidence.
        confidence = torch.abs(prediction.view(-1))
        loss = self.weight * torch.mean(contradiction_score * (confidence ** 2))
        return loss

# ---------------------------
# Component 7: TradingModel Wrapper
# ---------------------------
class TradingModel(nn.Module):
    def __init__(self, tech_input_dim, sentiment_input_dim, encoder_hidden_dim, proj_dim, decision_hidden_dim):
        """
        End-to-end model:
          1. Encode technical and sentiment inputs.
          2. Project into a shared latent space.
          3. Fuse adaptively using a contradiction-aware gate.
          4. Produce a final decision output.
        """
        super(TradingModel, self).__init__()
        self.encoder_tech = EncoderTechnical(tech_input_dim, encoder_hidden_dim, dropout_prob=0.1)
        self.encoder_sent = EncoderSentiment(sentiment_input_dim, encoder_hidden_dim, dropout_prob=0.1)
        self.projection = ProjectionHead(encoder_hidden_dim, proj_dim)
        self.adaptive_fusion = AdaptiveFusion(proj_dim)
        self.decision_head = DecisionHead(proj_dim, decision_hidden_dim, output_dim=1)
        
    def forward(self, tech_data, sentiment_data):
        # Step 1: Encode each modality.
        tech_features = self.encoder_tech(tech_data)
        sent_features = self.encoder_sent(sentiment_data)
        
        # Step 2: Project to shared latent space.
        proj_tech = self.projection(tech_features)
        proj_sent = self.projection(sent_features)
        
        # Step 3: Fuse using adaptive fusion.
        fused, contradiction_score, gate_weight = self.adaptive_fusion(proj_tech, proj_sent)
        
        # Step 4: Get final decision.
        decision = self.decision_head(fused)
        return decision, contradiction_score, proj_tech, proj_sent, gate_weight

# ---------------------------
# Testing and Training Loop Scaffold
# ---------------------------
if __name__ == '__main__':
    # Define dimensions for dummy data.
    batch_size = 8
    tech_input_dim = 10         # e.g., OHLCV + technical indicators.
    sentiment_input_dim = 768   # e.g., FinBERT embedding size.
    encoder_hidden_dim = 64
    proj_dim = 32
    decision_hidden_dim = 64
    
    # Create dummy inputs.
    tech_data = torch.randn(batch_size, tech_input_dim)
    sentiment_data = torch.randn(batch_size, sentiment_input_dim)
    # Dummy target: scalar for each sample.
    target = torch.randn(batch_size, 1)
    
    # Instantiate the model and loss modules.
    model = TradingModel(tech_input_dim, sentiment_input_dim, encoder_hidden_dim, proj_dim, decision_hidden_dim)
    prediction_loss_fn = nn.MSELoss()
    contradiction_loss_fn = ContradictionLoss(weight=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Training loop (using dummy data).
    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass.
        decision, contradiction_score, proj_tech, proj_sent, gate_weight = model(tech_data, sentiment_data)
        
        # Compute primary prediction loss.
        primary_loss = prediction_loss_fn(decision, target)
        # Compute contradiction regularization loss.
        contr_loss = contradiction_loss_fn(proj_tech, proj_sent, decision)
        
        total_loss = primary_loss + contr_loss
        total_loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1}: Total Loss = {total_loss.item():.4f}, Primary Loss = {primary_loss.item():.4f}, Contradiction Loss = {contr_loss.item():.4f}")
    
    # Testing a forward pass.
    model.eval()
    with torch.no_grad():
        decision, contradiction_score, proj_tech, proj_sent, gate_weight = model(tech_data, sentiment_data)
        print("Sample Decision Output:", decision)
        print("Sample Contradiction Score:", contradiction_score)
        print("Sample Gate Weights:", gate_weight)import numpy as np
import matplotlib.pyplot as plt
import argparse
from sklearn.metrics import accuracy_score

def compute_direction_accuracy(predictions, targets):
    pred_dir = predictions > 0
    target_dir = targets > 0
    return np.mean(pred_dir == target_dir)

def compute_sharpe_ratio(returns, risk_free_rate=0.0):
    excess = returns - risk_free_rate
    return excess.mean() / (excess.std() + 1e-8)

def evaluate_strategy(predictions, targets, contradiction_tags, target_mode):
    overall_dir_acc = compute_direction_accuracy(predictions, targets)
    overall_avg_return = np.mean(predictions)
    overall_sharpe = compute_sharpe_ratio(predictions)

    print("evaluate_strategy.py Debug:")
    print("  Overall baseline average target return:", np.mean(targets))

    unique_tags, counts = np.unique(contradiction_tags, return_counts=True)
    print("  Contradiction Tags Distribution:", dict(zip(unique_tags, counts)))

    per_type_metrics = {}
    for tag in unique_tags:
        mask = contradiction_tags == tag
        if np.sum(mask) > 0:
            if target_mode == "binary":
                binary_preds = predictions[mask] > 0.5
                binary_targets = targets[mask] == 1
                acc = accuracy_score(binary_targets, binary_preds)
            else:
                acc = compute_direction_accuracy(predictions[mask], targets[mask])
            avg_ret = np.mean(predictions[mask])
            sharpe = compute_sharpe_ratio(predictions[mask])
            per_type_metrics[tag] = (acc, avg_ret, sharpe)
            print(f"  Metrics for {tag}: Direction Accuracy = {acc:.2%}, Average Return = {avg_ret:.4f}, Sharpe Ratio = {sharpe:.4f}")

    return (overall_dir_acc, overall_avg_return, overall_sharpe), per_type_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_mode", type=str, default="normalized", choices=["normalized", "binary", "rolling"])
    parser.add_argument("--filter_underhype", action="store_true")
    parser.add_argument("--filter_overhype", action="store_true")
    parser.add_argument("--filter_none", action="store_true")
    args = parser.parse_args()

    target_mode = args.target_mode

    data = np.load("./training_data/predictions_routed.npz", allow_pickle=True)
    predictions = data["predictions"]
    targets = data["target_returns"]
    contradiction_tags = data["contradiction_tags"]

    # Conditional filtering
    if args.filter_underhype or args.filter_overhype or args.filter_none:
        if args.filter_underhype:
            filter_type = "underhype"
        elif args.filter_overhype:
            filter_type = "overhype"
        else:
            filter_type = "none"
        mask = contradiction_tags == filter_type
        predictions = predictions[mask]
        targets = targets[mask]
        contradiction_tags = contradiction_tags[mask]
        print(f"Evaluation filtering applied: {np.sum(~mask)} samples removed.")

    overall, per_type = evaluate_strategy(predictions, targets, contradiction_tags, target_mode)
    print("Overall Metrics:")
    print("  Direction Accuracy: {:.2%}".format(overall[0]))
    print("  Average Predicted Return: {:.4f}".format(overall[1]))
    print("  Sharpe Ratio: {:.4f}".format(overall[2]))

    plt.figure(figsize=(6, 4))
    unique, counts = np.unique(contradiction_tags, return_counts=True)
    plt.bar(unique, counts)
    plt.xlabel("Contradiction Type")
    plt.ylabel("Count")
    plt.title("Contradiction Type Distribution")
    plt.show()
import numpy as np
import os

def filter_and_save_by_tag(tag):
    dataset_path = "./training_data/dataset.npz"
    predictions_path = "./training_data/predictions.npz"
    output_path = f"./training_data/{tag}_only_dataset.npz"
    
    data = np.load(dataset_path)
    preds = np.load(predictions_path, allow_pickle=True)
    tags = preds["contradiction_tags"]
    
    indices = np.where(tags == tag)[0]
    print(f"{tag} samples:", len(indices))
    
    filtered = {key: data[key][indices] for key in data}
    np.savez(output_path, **filtered)
    print(f"✅ Saved {tag} dataset to {output_path}")

if __name__ == "__main__":
    os.makedirs("training_data", exist_ok=True)
    for tag in ["underhype", "overhype", "none"]:
        filter_and_save_by_tag(tag)
import numpy as np
import os

def filter_underhype_dataset(dataset_path, predictions_path, output_path):
    """
    Loads the full dataset and predictions, filters samples with contradiction_tag "underhype",
    and saves the filtered dataset.
    """
    data = np.load(dataset_path)
    preds = np.load(predictions_path, allow_pickle=True)
    tags = preds["contradiction_tags"]
    
    underhype_indices = np.where(tags == "underhype")[0]
    print("Number of underhype samples:", len(underhype_indices))
    
    filtered_data = {}
    for key in data.keys():
        filtered_data[key] = data[key][underhype_indices]
    
    np.savez(output_path, **filtered_data)
    print("Filtered dataset saved to", output_path)

if __name__ == "__main__":
    dataset_path = "./training_data/dataset.npz"
    predictions_path = "./training_data/predictions.npz"
    output_path = "./training_data/underhype_only_dataset.npz"
    os.makedirs("./training_data", exist_ok=True)
    filter_underhype_dataset(dataset_path, predictions_path, output_path)from transformers import AutoTokenizer, AutoModel
import fin_dataset

def embeddings(tokenizer, model, text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    cls_embed = outputs.last_hidden_state[:, 0, :].detach().numpy()
    return cls_embed

def start_load(model_name="yiyanghkust/finbert-tone"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
    return tokenizer, model


if __name__ == "__main__":
    phrasebank_data = fin_dataset.load_phrasebank()
    
    tokenizer, model = start_load("yiyanghkust/finbert-tone")
    
    sample_text = phrasebank_data[0]["sentence"]
    embedding = embeddings(tokenizer, model, sample_text)
    
    print("sample sentence:", sample_text)
    print("CLS embedding shape:", embedding.shape)import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def init_weights(m):
    """Initialize weights for linear layers using Xavier uniform initialization."""
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class FusionNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, output_dim=1, use_attention=False, fusion_method='average', target_mode="normalized", dropout_prob=0.5, logit_clamp_min=-10, logit_clamp_max=10):
        """
        fusion_method: 'average' or 'concat'
            - For 'concat', fc1 input dimension becomes input_dim + 768.
            - For 'average', we simply average emb1 and emb2 and fc1 input is input_dim.
        target_mode: "normalized", "binary", or "rolling". Determines loss function and output activation.
            - For "binary", the model outputs raw logits during training and applies clamped sigmoid during evaluation.
            - For "normalized" and "rolling", the model outputs a scalar regression value.
        dropout_prob: Dropout probability applied after fc1 and fc2. Dropout remains active (MC Dropout) even during evaluation for uncertainty estimation.
        logit_clamp_min, logit_clamp_max: Clamp values for logits in binary mode during evaluation.
        """
        super(FusionNet, self).__init__()
        self.use_attention = use_attention
        self.fusion_method = fusion_method
        self.target_mode = target_mode
        self.dropout_prob = dropout_prob
        self.logit_clamp_min = logit_clamp_min
        self.logit_clamp_max = logit_clamp_max
        
        if fusion_method == 'concat':
            fc1_in_dim = input_dim + 768
        elif fusion_method == 'average':
            fc1_in_dim = input_dim
        else:
            raise ValueError("Invalid fusion_method; choose 'concat' or 'average'.")
        
        self.fc1 = nn.Linear(fc1_in_dim, hidden_dim)
        if self.use_attention:
            self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=1, batch_first=True)
        else:
            self.attention = None
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        
        self.apply(init_weights)
    
    def forward(self, emb1, emb2):
        # Fusion: average or concatenate.
        if self.fusion_method == 'average':
            x = (emb1 + emb2) / 2.0
        else:
            x = torch.cat((emb1, emb2), dim=1)
        
        if torch.isnan(x).any():
            print("NaN detected after fusion, x shape:", x.shape)
        
        # First FC layer + ReLU.
        h = self.fc1(x)
        h = self.relu(h)
        # MC Dropout: always active.
        h = F.dropout(h, p=self.dropout_prob, training=True)
        
        # Optional attention mechanism.
        if self.use_attention:
            h_seq = h.unsqueeze(1)
            if torch.isnan(h_seq).any():
                print("NaN detected before attention (h_seq)")
            attn_output, _ = self.attention(h_seq, h_seq, h_seq)
            h = attn_output.squeeze(1)
            if torch.isnan(h).any():
                print("NaN detected after attention squeeze")
        
        # Second FC layer + ReLU.
        h2 = self.fc2(h)
        h2 = self.relu(h2)
        h2 = F.dropout(h2, p=self.dropout_prob, training=True)
        if torch.isnan(h2).any():
            print("NaN detected after fc2 and dropout")
        
        # Final output layer.
        output = self.out(h2)
        if torch.isnan(output).any():
            print("NaN detected after output layer")
        
        # For binary mode during evaluation, clamp logits and apply sigmoid.
        if self.target_mode == "binary" and not self.training:
            output = torch.clamp(output, min=self.logit_clamp_min, max=self.logit_clamp_max)
            output = torch.sigmoid(output)
        
        return output.view(-1)

    def predict(self, x1, x2):
        self.eval()
        if not isinstance(x1, torch.Tensor):
            x1 = torch.from_numpy(x1.astype(np.float32))
        if not isinstance(x2, torch.Tensor):
            x2 = torch.from_numpy(x2.astype(np.float32))
        with torch.no_grad():
            y = self.forward(x1.unsqueeze(0), x2.unsqueeze(0))
        return y.squeeze(0).cpu().numpy()

    def save_model(self, filepath):
        torch.save(self.state_dict(), filepath)
    
    def load_model(self, filepath):
        self.load_state_dict(torch.load(filepath))
        self.eval()

if __name__ == "__main__":
    # Quick test.
    dummy_emb1 = torch.randn(32, 10)
    dummy_emb2 = torch.randn(32, 768)
    model = FusionNet(input_dim=10, hidden_dim=512, use_attention=True, fusion_method='concat', target_mode="binary")
    output = model(dummy_emb1, dummy_emb2)
    print("Test output:", output)import networkx as nx
import numpy as np
import pandas as pd
import pandas_ta as ta
import yfinance as yf
import plotly.graph_objects as go
import torch
from fusionnet import FusionNet, neural_fusion  # Fusion network and neural fusion function
from fin_dataset import load_phrasebank         # Function to load the news dataset
from finbert import embeddings, start_load         # FinBERT helper functions
import math

# === Placeholder functions for external data ===
def fetch_options_chain_features(ticker, date):
    """Fetch option chain data for a given ticker and date."""
    return {
        "implied_volatility": 0.2,
        "put_call_ratio": 0.7,
        "open_interest_call": 10000,
        "open_interest_put": 8000
    }

def fetch_macro_indicators(date):
    """Fetch macroeconomic indicators for a given date."""
    return {
        "VIX": 18.5,
        "FED_FUNDS_RATE": 0.5,
        "GDP_GROWTH": 0.02
    }

# === Risk filter ===
def risk_filter(market_data, portfolio):
    """
    Check if current market conditions pass risk filters.
    For example, skip trades if VIX is very high or if drawdown > 20%.
    """
    vix = market_data.get("VIX", None)
    if vix is not None and vix > 30:
        return False
    if portfolio.get('max_drawdown', 0) > 0.2:
        return False
    return True

# === TradingGraph class ===
class TradingGraph:
    """
    Graph-based trading system that builds nodes from market and news data,
    detects contradictory signals, fuses them with a neural network,
    and simulates trades with backtesting and live simulation scaffolding.
    """
    def __init__(self, model, initial_capital=100000, slippage=0.001, fee_per_trade=1.0, use_options=False):
        self.model = model
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions = []    
        self.trade_log = []    
        self.equity_curve = [] 
        self.benchmark_curve = []  
        self.slippage = slippage
        self.fee_per_trade = fee_per_trade
        self.use_options = use_options
        self.max_drawdown = 0.0
        self.peak_equity = initial_capital

    def _apply_slippage(self, price, side):
        if side.lower() == 'buy':
            return price * (1 + self.slippage)
        elif side.lower() == 'sell':
            return price * (1 - self.slippage)
        else:
            return price

    def _apply_fee(self):
        self.capital -= self.fee_per_trade

    def _update_equity_curve(self, current_price, benchmark_price=None):
        total_value = self.capital
        for pos in self.positions:
            if pos.get('type') == 'option':
                total_value += pos['quantity'] * pos['price']
            else:
                total_value += pos['quantity'] * current_price
        self.equity_curve.append(total_value)
        if total_value > self.peak_equity:
            self.peak_equity = total_value
        drawdown = (self.peak_equity - total_value) / self.peak_equity
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown
        if benchmark_price is not None:
            if not self.benchmark_curve:
                self.benchmark_shares = self.initial_capital / benchmark_price
            benchmark_value = self.benchmark_shares * benchmark_price
            self.benchmark_curve.append(benchmark_value)
        else:
            if not self.benchmark_curve:
                self.benchmark_shares = self.initial_capital / current_price
            benchmark_value = self.benchmark_shares * current_price
            self.benchmark_curve.append(benchmark_value)

    def generate_features(self, market_data):
        features = []
        price = market_data.get('price')
        volume = market_data.get('volume')
        features.extend([price, volume])
        if 'indicators' in market_data:
            features.extend(list(market_data['indicators'].values()))
        if 'options' in market_data:
            opt_feats = market_data['options']
        else:
            opt_feats = fetch_options_chain_features(market_data.get('ticker', ''), market_data.get('date', None))
        features.extend(list(opt_feats.values()))
        if 'macro' in market_data:
            macro_feats = market_data['macro']
        else:
            macro_feats = fetch_macro_indicators(market_data.get('date', None))
        features.extend(list(macro_feats.values()))
        return np.array(features, dtype=float)

    def decide_trade(self, model_output, current_price, current_date=None):
        signal = model_output
        if signal > 0:
            direction = 'buy'
        elif signal < 0:
            direction = 'sell'
        else:
            direction = 'hold'
        confidence = min(1.0, abs(signal))
        base_position_size = 1  
        position_size = base_position_size * confidence
        trade_type = 'directional'
        if self.use_options and confidence < 0.5:
            trade_type = 'non-directional'
        return direction, position_size, trade_type

    def execute_trade(self, direction, size, current_price, trade_type='directional'):
        if direction == 'hold' or size == 0:
            return
        if direction == 'buy':
            trade_sign = 1
        elif direction == 'sell':
            trade_sign = -1
        else:
            trade_sign = 0
        executed_price = self._apply_slippage(current_price, direction)
        trade_cost = executed_price * size * trade_sign * -1  
        self.capital += trade_cost
        self._apply_fee()
        if trade_sign == 1:
            position = {
                'type': 'option' if self.use_options and trade_type != 'directional' else 'asset',
                'quantity': size,
                'entry_price': executed_price,
                'price': executed_price
            }
            self.positions.append(position)
        elif trade_sign == -1:
            if self.positions:
                self.positions.pop(0)
        self.trade_log.append({
            'date': None,
            'direction': direction,
            'size': size,
            'price': executed_price,
            'trade_type': trade_type,
            'capital_after': self.capital
        })

    def apply_time_decay(self):
        if self.use_options:
            for pos in self.positions:
                if pos.get('type') == 'option':
                    decay_rate = 0.001
                    pos['price'] = pos['price'] * (1 - decay_rate)

    def backtest(self, historical_data, benchmark_prices=None):
        for t, data_point in enumerate(historical_data):
            current_price = data_point.get('price')
            current_date = data_point.get('date', None)
            features = self.generate_features(data_point)
            model_output = self.model.predict(features)
            safe_to_trade = risk_filter({**data_point, **data_point.get('macro', {})}, 
                                        {'max_drawdown': self.max_drawdown})
            if not safe_to_trade:
                self._update_equity_curve(current_price, benchmark_price=(benchmark_prices[t] if benchmark_prices is not None else None))
                continue
            direction, size, trade_type = self.decide_trade(model_output, current_price, current_date)
            self.execute_trade(direction, size, current_price, trade_type)
            self.apply_time_decay()
            bench_price = benchmark_prices[t] if benchmark_prices is not None else None
            self._update_equity_curve(current_price, benchmark_price=bench_price)
        return {
            "equity_curve": self.equity_curve,
            "benchmark_curve": self.benchmark_curve,
            "trade_log": self.trade_log,
            "max_drawdown": self.max_drawdown
        }

    def live_run(self, data_stream, benchmark_symbol=None):
        print("Starting live simulation... (placeholder)")
        for data_point in data_stream:
            current_price = data_point.get('price')
            features = self.generate_features(data_point)
            model_output = self.model.predict(features)
            if not risk_filter({**data_point, **data_point.get('macro', {})}, {'max_drawdown': self.max_drawdown}):
                continue
            direction, size, trade_type = self.decide_trade(model_output, current_price)
            self.execute_trade(direction, size, current_price, trade_type)
            self.apply_time_decay()
            self._update_equity_curve(current_price)
            if direction != 'hold' and size > 0:
                print(f"Executed {direction} trade of size {size} at price {current_price:.2f}.")
        print("Live simulation ended. (placeholder for real-time trading)")

# === Utility functions ===
def load_market_data(ticker, start, end):
    return yf.download(ticker, start=start, end=end)

def add_technical_indicators(data):
    data["SMA20"] = ta.sma(data["Close"], length=20)
    data["RSI"] = ta.rsi(data["Close"], length=14)
    return data

def adaptive_detect_contradiction(e1, e2, tech_diff=0.0, sentiment_weight=0.69, tech_weight=0.31, threshold=0.5):
    from sklearn.metrics.pairwise import cosine_similarity
    sim = cosine_similarity(e1.reshape(1, -1), e2.reshape(1, -1))[0][0]
    sentiment_metric = 1.0 - sim 
    combined_metric = sentiment_weight * sentiment_metric + tech_weight * tech_diff
    return combined_metric > threshold, combined_metric

def create_combined_state(date, market_data, news_text, tokenizer, model, options_data=None, macro_data=None, news_sentiment=None):
    sentiment_emb = embeddings(tokenizer, model, news_text)
    try:
        row = market_data.loc[date]
    except Exception as e:
        print(f"Error: Date {date} not found in market data. Using last available row.")
        row = market_data.iloc[-1]
    technical_info = {
        "Close": row["Close"],
        "Open": row.get("Open", None),
        "High": row.get("High", None),
        "Low": row.get("Low", None),
        "Volume": row.get("Volume", None),
        "SMA20": row.get("SMA20", None),
        "SMA50": row.get("SMA50", None),
        "EMA20": row.get("EMA20", None),
        "EMA50": row.get("EMA50", None),
        "RSI14": row.get("RSI14", None),
        "MACD": row.get("MACD", None),
        "StochK": row.get("StochK", None),
        "StochD": row.get("StochD", None),
        "HistoricalVol20": market_data["Close"].pct_change().rolling(window=20).std().iloc[-1] if "Close" in market_data.columns else None,
        "ATR14": row.get("ATR14", None),
        "ImpliedVol": None  # Placeholder for options data
    }
    state = {
        "date": date,
        "news_text": news_text,
        "sentiment_embedding": sentiment_emb,
        "technical": technical_info
    }
    return state

def options_trading_signal(enriched_state, uncertainty=0.0, sentiment_threshold=0.0, uncertainty_threshold=0.5):
    sentiment_mean = enriched_state.mean()
    if sentiment_mean > sentiment_threshold and uncertainty < uncertainty_threshold:
        return "buy_call"
    elif sentiment_mean < -sentiment_threshold and uncertainty < uncertainty_threshold:
        return "buy_put"
    else:
        return "straddle"

def backtest_signals(prices, signals):
    position = 0
    returns = []
    for i in range(1, len(prices)):
        if signals[i-1] == "buy_call" and position == 0:
            entry = prices.iloc[i-1]
            position = 1
        if signals[i-1] == "buy_put" and position == 1:
            exit_price = prices.iloc[i]
            returns.append(exit_price / entry - 1)
            position = 0
    if returns:
        return np.prod([1 + r for r in returns]) - 1
    else:
        return 0

def visualize_graph_interactive(G):
    pos = nx.spring_layout(G)
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1, color='#888'), hoverinfo='none', mode='lines')
  
    node_x, node_y, text = [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        state = G.nodes[node].get("state", {})
        snippet = state.get("news_text", "")[:40] + "..." if state.get("news_text") else ""
        sig = options_trading_signal(state.get("sentiment_embedding")) if state.get("sentiment_embedding") is not None else ""
        text.append(f"Node {node}: {snippet}<br>Signal: {sig}")
  
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=text,
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            reversescale=True,
            color=[],  # You can assign colors based on signal intensity, drawdown, etc.
            size=10,
            colorbar=dict(thickness=15, title='Node Connections', xanchor='left'),
            line_width=2
        )
    )
  
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='<br>Interactive Knowledge Graph',
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        annotations=[dict(text="", showarrow=False, xref="paper", yref="paper")],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                    ))
    fig.show()

# === Main function ===
def main():
    # Load market data for ticker "AAPL" between given dates.
    ticker = "AAPL"
    market = load_market_data(ticker, "2023-01-01", "2023-01-31")
    market = add_technical_indicators(market)
    print("Market data loaded. Shape:", market.shape)
  
    # Load news dataset using your function.
    news_dataset = load_phrasebank()
    print("News dataset length:", len(news_dataset))
  
    # Load FinBERT tokenizer and model.
    tokenizer, model = start_load(model_name="yiyanghkust/finbert-tone")
  
    G = nx.DiGraph()
    options_signals = []
    node_index = 0
    # Convert market dates to list of strings.
    dates = market.index.strftime("%Y-%m-%d").tolist()
  
    # Create nodes from news dataset. (Limit to 5 nodes for now.)
    for i, sample in enumerate(news_dataset):
        if isinstance(sample, dict) and sample.get("sentence", "").strip().lower() == "sentence":
            continue
        news_text = sample.get("sentence", "") if isinstance(sample, dict) else sample
        date = dates[i] if i < len(dates) else dates[-1]
        state = create_combined_state(date, market, news_text, tokenizer, model)
        G.add_node(node_index, state=state)
        sig = options_trading_signal(state["sentiment_embedding"])
        options_signals.append(sig)
        print(f"Node {node_index} added for date {date} with signal: {sig}")
        node_index += 1
        if node_index >= 5:
            break

    # Create simple sequential edges between nodes.
    for i in range(len(G.nodes) - 1):
        G.add_edge(i, i+1, transformation="sequential")
  
    # Check for contradiction between first two nodes and fuse if needed.
    if len(G.nodes) >= 2:
        emb0 = G.nodes[0]['state']["sentiment_embedding"]
        emb1 = G.nodes[1]['state']["sentiment_embedding"]
        if emb0.ndim == 1:
            emb0 = emb0.reshape(1, -1)
        if emb1.ndim == 1:
            emb1 = emb1.reshape(1, -1)
        tech_diff = 0.3  # Example technical difference; replace with actual computed value.
        contradiction_flag, combined_metric = adaptive_detect_contradiction(emb0, emb1, tech_diff=tech_diff)
        print(f"Adaptive contradiction detected between node 0 and 1: {contradiction_flag} (metric: {combined_metric:.2f})")
        if contradiction_flag:
            fusion_net = FusionNet(input_dim=768, hidden_dim=512, output_dim=768, use_attention=True)
            try:
                fusion_net.load_state_dict(torch.load("fusion_net_weights.pth"))
                fusion_net.eval()
                print("Loaded fusion_net_weights.pth")
            except Exception as e:
                print("No pre-trained fusion net weights found; using untrained model.")
            fused_embedding, uncertainty = neural_fusion(emb0, emb1, fusion_net)
            print("Neurally fused embedding shape:", fused_embedding.shape)
            print("Fusion uncertainty measure:", uncertainty)
            fused_signal = options_trading_signal(fused_embedding, uncertainty)
            print("Fused options trading signal:", fused_signal)
  
    print("Options trading signals:", options_signals)
    prices = market["Close"].iloc[:len(options_signals)]
    cum_return = backtest_signals(prices, options_signals)
    print("Cumulative return from backtesting:", cum_return)
  
    visualize_graph_interactive(G)
  
    # Print each node's summary.
    for node in G.nodes(data=True):
        state = node[1]["state"]
        snippet = state.get("news_text", "")[:40] + "..."
        print(f"Node {node[0]}: Date: {state['date']}, News Snippet: {snippet}, Signal: {options_trading_signal(state['sentiment_embedding'])}")

if __name__ == "__main__":
    main()import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionFusion(nn.Module):
    def __init__(self, tech_dim, semantic_dim, fusion_dim, num_heads=4):
        """
        Fusion layer that uses multi-head attention.
        Projects technical and semantic inputs into a shared space, computes attention,
        and outputs a fused representation.
        """
        super(AttentionFusion, self).__init__()
        self.tech_proj = nn.Linear(tech_dim, fusion_dim)
        self.semantic_proj = nn.Linear(semantic_dim, fusion_dim)
        self.attention = nn.MultiheadAttention(embed_dim=fusion_dim, num_heads=num_heads, batch_first=True)
        # A gating mechanism to modulate fusion based on contradiction severity.
        self.gate = nn.Sequential(
            nn.Linear(1, fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, fusion_dim),
            nn.Sigmoid()
        )
    
    def forward(self, tech_input, semantic_input, contradiction_severity):
        """
        Args:
            tech_input: Tensor of shape [batch, tech_dim]
            semantic_input: Tensor of shape [batch, semantic_dim]
            contradiction_severity: Tensor of shape [batch] (e.g., 1 - cosine similarity)
        Returns:
            fused: Tensor of shape [batch, fusion_dim]
        """
        batch_size = tech_input.size(0)
        tech_proj = self.tech_proj(tech_input)         # [batch, fusion_dim]
        semantic_proj = self.semantic_proj(semantic_input)  # [batch, fusion_dim]
        
        # Concatenate along a "sequence" dimension (treat as two tokens).
        fused_tokens = torch.stack([tech_proj, semantic_proj], dim=1)  # [batch, 2, fusion_dim]
        
        # Compute multi-head attention using the tokens themselves as query, key, value.
        attn_output, _ = self.attention(fused_tokens, fused_tokens, fused_tokens)
        # Average the two tokens.
        fused = attn_output.mean(dim=1)  # [batch, fusion_dim]
        
        # Gate modulation based on contradiction severity.
        contradiction_severity = contradiction_severity.unsqueeze(1)  # [batch, 1]
        gate_modulation = self.gate(contradiction_severity)           # [batch, fusion_dim]
        fused = fused * gate_modulation  # Elementwise modulation.
        
        return fused

if __name__ == "__main__":
    # Quick test of the AttentionFusion module.
    batch_size = 8
    tech_dim = 10
    semantic_dim = 768
    fusion_dim = 128
    dummy_tech = torch.randn(batch_size, tech_dim)
    dummy_semantic = torch.randn(batch_size, semantic_dim)
    # Simulated contradiction severity (e.g., higher means more contradiction).
    dummy_severity = torch.rand(batch_size)
    
    fusion_module = AttentionFusion(tech_dim, semantic_dim, fusion_dim)
    fused_output = fusion_module(dummy_tech, dummy_semantic, dummy_severity)
    print("Fused output shape:", fused_output.shape)# kfold_cross_validation.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from fusionnet import FusionNet
from contradiction_engine import AdaptiveContradictionEngine
from sklearn.model_selection import KFold

def compute_direction_accuracy(predictions, targets):
    pred_dir = predictions > 0
    target_dir = targets > 0
    return (pred_dir == target_dir).mean()

def compute_sharpe_ratio(returns, risk_free_rate=0.0):
    excess = returns - risk_free_rate
    return excess.mean() / (excess.std() + 1e-8)

def run_kfold_cv(dataset_path, n_splits=5, num_epochs=5):
    data = np.load(dataset_path)
    tech_data = data["technical_features"]
    finbert_data = data["finbert_embeddings"]
    price_data = data["price_movements"]
    sentiment_data = data["news_sentiment_scores"]
    target_returns = data["target_returns"]
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    metrics = []
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for train_index, test_index in kf.split(tech_data):
        # Prepare train/test tensors.
        tech_train = torch.tensor(tech_data[train_index], dtype=torch.float32).to(device)
        finbert_train = torch.tensor(finbert_data[train_index], dtype=torch.float32).to(device)
        price_train = torch.tensor(price_data[train_index], dtype=torch.float32).to(device)
        sentiment_train = torch.tensor(sentiment_data[train_index], dtype=torch.float32).to(device)
        target_train = torch.tensor(target_returns[train_index], dtype=torch.float32).to(device)
        
        tech_test = torch.tensor(tech_data[test_index], dtype=torch.float32).to(device)
        finbert_test = torch.tensor(finbert_data[test_index], dtype=torch.float32).to(device)
        price_test = torch.tensor(price_data[test_index], dtype=torch.float32).to(device)
        sentiment_test = torch.tensor(sentiment_data[test_index], dtype=torch.float32).to(device)
        target_test = torch.tensor(target_returns[test_index], dtype=torch.float32).to(device)
        
        # Initialize model and contradiction engine.
        model = FusionNet(tech_input_dim=10, hidden_dim=512, use_attention=True, fusion_method='concat').to(device)
        contradiction_engine = AdaptiveContradictionEngine(embedding_dim=768).to(device)
        loss_fn = nn.MSELoss()
        optimizer = optim.Adam(list(model.parameters()) + list(contradiction_engine.parameters()), lr=1e-3)
        
        # Simple training loop.
        for epoch in range(num_epochs):
            model.train()
            permutation = torch.randperm(tech_train.size(0))
            epoch_loss = 0.0
            for i in range(0, tech_train.size(0), 128):
                indices = permutation[i:i+128]
                batch_tech = tech_train[indices]
                batch_finbert = finbert_train[indices]
                batch_price = price_train[indices]
                batch_sentiment = sentiment_train[indices]
                batch_target = target_train[indices]
                
                optimizer.zero_grad()
                updated_embeddings = []
                for j in range(batch_finbert.size(0)):
                    updated_emb, _ = contradiction_engine(batch_finbert[j], batch_tech[j], batch_price[j], batch_sentiment[j])
                    updated_embeddings.append(updated_emb)
                updated_embeddings = torch.stack(updated_embeddings)
                pred = model(batch_tech, updated_embeddings).view(-1)
                loss = loss_fn(pred, batch_target.view(-1))
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch_tech.size(0)
            #print(f"Fold training epoch {epoch+1}: Loss = {epoch_loss/tech_train.size(0):.4f}")
        
        # Evaluation on test fold.
        model.eval()
        with torch.no_grad():
            updated_embeddings_test = []
            for j in range(finbert_test.size(0)):
                updated_emb, _ = contradiction_engine(finbert_test[j], tech_test[j], price_test[j], sentiment_test[j])
                updated_embeddings_test.append(updated_emb)
            updated_embeddings_test = torch.stack(updated_embeddings_test)
            predictions = model(tech_test, updated_embeddings_test).view(-1)
            predictions_np = predictions.cpu().numpy()
            targets_np = target_test.view(-1).cpu().numpy()
            dir_acc = compute_direction_accuracy(predictions_np, targets_np)
            avg_ret = predictions_np.mean()
            sharpe = compute_sharpe_ratio(predictions_np)
            metrics.append((dir_acc, avg_ret, sharpe))
            print(f"Fold metrics: Direction Acc: {dir_acc:.2%}, Avg Return: {avg_ret:.4f}, Sharpe: {sharpe:.4f}")
    return metrics

if __name__ == "__main__":
    metrics = run_kfold_cv("./training_data/dataset.npz", n_splits=5, num_epochs=5)
    metrics = np.array(metrics)
    print("Average metrics across folds:")
    print("Direction Accuracy: {:.2%}".format(metrics[:,0].mean()))
    print("Average Return: {:.4f}".format(metrics[:,1].mean()))
    print("Sharpe Ratio: {:.4f}".format(metrics[:,2].mean()))#!/usr/bin/env python3
import argparse
import ast
import numpy as np
import pandas as pd
import torch
import yfinance as yf
import ta  # Technical analysis library (install via pip install ta)
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import feedparser

# Import your TradingModel from trading_model.py (which encapsulates EncoderTechnical, EncoderSentiment, etc.)
from trading_model import TradingModel  # Ensure this file is available in your project

# Load FinBERT tokenizer and model once.
finbert_tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
finbert_model = AutoModel.from_pretrained("yiyanghkust/finbert-tone")
finbert_model.eval()

def get_finbert_cls_embedding(text: str) -> np.ndarray:
    inputs = finbert_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = finbert_model(**inputs)
    # Return the CLS token embedding (first token) as a numpy array.
    cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
    return cls_embedding

def get_live_headlines(ticker: str) -> list:
    # Attempt to fetch live news headlines from Yahoo Finance RSS feed
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
    feed = feedparser.parse(url)
    headlines = [entry["title"] for entry in feed.entries[:5]]
    
    if not headlines:
        # Fallback headlines if feed is empty
        headlines = [
            f"{ticker} reports record quarterly earnings amid market turbulence",
            f"Analysts see growth potential in {ticker} despite global uncertainties"
        ]
    return headlines

def compute_technical_indicators(ohlcv: pd.DataFrame) -> pd.DataFrame:
    # Use ta library to compute RSI, EMA, and MACD.
    ohlcv = ohlcv.copy()
    ohlcv["RSI"] = ta.momentum.rsi(ohlcv["Close"], window=14)
    ohlcv["EMA"] = ta.trend.ema_indicator(ohlcv["Close"], window=20)
    macd = ta.trend.macd(ohlcv["Close"], window_slow=26, window_fast=12)
    ohlcv["MACD"] = macd
    # Fill missing values.
    ohlcv.fillna(method="ffill", inplace=True)
    return ohlcv

def get_features(ticker: str) -> (np.ndarray, np.ndarray, float):
    # Fetch live OHLCV data for the last 60 days.
    data = yf.download(ticker, period="60d", interval="1d")
    if data.empty:
        raise ValueError(f"No data returned for ticker {ticker}.")
    data = compute_technical_indicators(data)
    # Use the latest row to compute technical features.
    latest = data.iloc[-1]
    # For demonstration, select a subset: [Open, High, Low, Close, Volume, RSI, EMA, MACD]
    tech_features = np.array([
        latest["Open"], latest["High"], latest["Low"], latest["Close"],
        latest["Volume"], latest["RSI"], latest["EMA"], latest["MACD"]
    ], dtype=np.float32)
    # If your model expects a fixed size (e.g., 10 features), pad with zeros.
    if tech_features.shape[0] < 10:
        tech_features = np.concatenate([tech_features, np.zeros(10 - tech_features.shape[0], dtype=np.float32)])
    # Get current price from the latest close.
    current_price = latest["Close"]
    return tech_features, data, current_price

def main():
    parser = argparse.ArgumentParser(description="Live prediction for contradiction-aware trading model.")
    parser.add_argument("--ticker", type=str, required=True, help="Ticker symbol (e.g., AAPL)")
    parser.add_argument("--current_price", type=float, default=None, help="Current asset price (if not provided, uses latest close)")
    parser.add_argument("--target_mode", type=str, default="normalized", choices=["normalized", "binary", "rolling"], help="Target mode used in training")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    # Load live OHLCV data and compute technical features.
    tech_features, ohlcv, current_price = get_features(args.ticker)
    if args.current_price is not None:
        current_price = args.current_price
    print(f"Current price for {args.ticker}: {current_price}")
    
    # Pull live news headlines.
    headlines = get_live_headlines(args.ticker)
    print("Live headlines:", headlines)
    
    # Get FinBERT embedding by concatenating headlines.
    combined_text = " ".join(headlines)
    finbert_embedding = get_finbert_cls_embedding(combined_text)
    print("FinBERT embedding (first 5 dims):", finbert_embedding[:5])
    
    # Normalize features if needed.
    # (Assume any necessary normalization/scaling is handled within your TradingModel or preprocessor.)
    tech_tensor = torch.tensor(tech_features, dtype=torch.float32).to(device)
    finbert_tensor = torch.tensor(finbert_embedding, dtype=torch.float32).to(device)
    
    # Load pretrained TradingModel.
    model_args = {
        "tech_input_dim": 10,
        "sentiment_input_dim": 768,
        "encoder_hidden_dim": 64,
        "proj_dim": 32,
        "decision_hidden_dim": 64
    }
    model = TradingModel(**model_args).to(device)
    model.load_state_dict(torch.load("models/trading_model.pth", map_location=device))
    model.eval()
    
    # Use torch.no_grad for inference.
    with torch.no_grad():
        # Here we assume the model accepts two inputs: technical and sentiment embeddings.
        # If your architecture requires additional processing, modify accordingly.
        decision, contradiction_score, *_ = model(tech_tensor.unsqueeze(0), finbert_tensor.unsqueeze(0))
        prediction = decision.view(-1).item()
    
    print("Predicted next-day return:", prediction)
    
    # Map predicted return to a projected price change.
    projected_price = current_price * (1 + prediction)
    print(f"Projected price based on predicted return: {projected_price:.2f}")
    
if __name__ == "__main__":
    main()#!/usr/bin/env python3
"""
live_router.py

Live signal generator that:
  - Accepts technical features, FinBERT embedding, price movement, sentiment score, and current price.
  - Uses the ContradictionEngine to tag the sample.
  - Routes the sample to the appropriate specialized FusionNet model.
  - Outputs the predicted return and calculates the projected target price.
Usage:
  python live_router.py --tech "[0.1,0.2,...,0.3]" --finbert "[0.1,0.2,...,0.5]" --price 0.012 --sentiment 0.7 --current_price 150 --target_mode normalized
"""
import argparse
import ast
import numpy as np
import torch
from fusionnet import FusionNet
from contradiction_engine import ContradictionEngine

def parse_args():
    parser = argparse.ArgumentParser(description="Live router for contradiction-aware trading signal generation.")
    parser.add_argument("--tech", type=str, required=True, help="Technical features as a JSON list string (e.g., '[0.1,0.2,...]')")
    parser.add_argument("--finbert", type=str, required=True, help="FinBERT embedding as a JSON list string (length 768)")
    parser.add_argument("--price", type=float, required=True, help="Price movement (e.g., 0.01 for 1% increase)")
    parser.add_argument("--sentiment", type=float, required=True, help="News sentiment score (e.g., 0.7)")
    parser.add_argument("--current_price", type=float, required=True, help="Current asset price")
    parser.add_argument("--target_mode", type=str, default="normalized", choices=["normalized", "binary", "rolling"], help="Target mode used in training")
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    # Parse inputs.
    tech_features = torch.tensor(ast.literal_eval(args.tech), dtype=torch.float32).to(device)
    finbert_embedding = torch.tensor(ast.literal_eval(args.finbert), dtype=torch.float32).to(device)
    price_movement = torch.tensor(args.price, dtype=torch.float32).to(device)
    sentiment_score = torch.tensor(args.sentiment, dtype=torch.float32).to(device)
    current_price = args.current_price

    # Load ContradictionEngine.
    contr_engine = ContradictionEngine(embedding_dim=768).to(device)
    contr_engine.eval()
    updated_emb, ctype = contr_engine(finbert_embedding, tech_features, price_movement, sentiment_score)
    if ctype is None:
        ctype = "none"
    print("Contradiction tag determined:", ctype)
    
    # Load specialized models.
    model_underhype = FusionNet(input_dim=len(tech_features), hidden_dim=512, use_attention=True, fusion_method='concat', target_mode=args.target_mode).to(device)
    model_overhype = FusionNet(input_dim=len(tech_features), hidden_dim=512, use_attention=True, fusion_method='concat', target_mode=args.target_mode).to(device)
    model_none = FusionNet(input_dim=len(tech_features), hidden_dim=512, use_attention=True, fusion_method='concat', target_mode=args.target_mode).to(device)
    model_underhype.load_state_dict(torch.load("./training_data/fusion_underhype_weights.pth", map_location=device))
    model_overhype.load_state_dict(torch.load("./training_data/fusion_overhype_weights.pth", map_location=device))
    model_none.load_state_dict(torch.load("./training_data/fusion_none_weights.pth", map_location=device))
    model_underhype.eval()
    model_overhype.eval()
    model_none.eval()
    
    # Route sample.
    tech_input = tech_features.unsqueeze(0)     # [1, d_tech]
    finbert_input = finbert_embedding.unsqueeze(0) # [1, 768]
    if ctype == "underhype":
        prediction = model_underhype(tech_input, finbert_input)
    elif ctype == "overhype":
        prediction = model_overhype(tech_input, finbert_input)
    else:
        prediction = model_none(tech_input, finbert_input)
    
    prediction = prediction.item()
    # Calculate projected price.
    target_price = current_price * (1 + prediction)
    
    # Determine action.
    threshold = 0.01
    if ctype == "underhype" and prediction > threshold:
        action = "LONG"
    elif ctype == "overhype" and prediction < -threshold:
        action = "SHORT"
    else:
        action = "NO_ACTION"
    
    print("Live Signal Output:")
    print("  Contradiction Type:", ctype)
    print("  Predicted Return:", prediction)
    print("  Projected Target Price:", target_price)
    print("  Action:", action)
    print("  Confidence (abs(prediction)):", abs(prediction))
    
if __name__ == "__main__":
    main()#!/usr/bin/env python3
import time
import logging
import argparse
import numpy as np
import torch
import joblib
import yfinance as yf
from datetime import datetime
from fusionnet import FusionNet
from contradiction_engine import ContradictionEngine

# Setup logging.
logging.basicConfig(
    filename="live_trading.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
console.setFormatter(formatter)
logging.getLogger().addHandler(console)

# -------------------------
# CONFIGURABLE PARAMETERS
# -------------------------
TICKER = "AAPL"
POSITION_SIZE = 50.0  # Dollars per trade
STOP_LOSS = 0.03      # 3% stop-loss
TAKE_PROFIT = 0.05    # 5% take-profit
TARGET_MODE = "normalized"  # or "binary" or "rolling"
DATA_FETCH_INTERVAL = 60 * 60 * 24  # run daily (in seconds)
MODEL_PATH = "./training_data/fusion_net_contradiction_weights.pth"
SCALER_PATH = "./training_data/target_scaler.pkl"  # Only used for normalized regression

# -------------------------
# MOCK FUNCTIONS (Replace with real implementations as needed)
# -------------------------
def fetch_market_data(ticker):
    """
    Fetches the latest daily OHLCV data using yfinance.
    Returns a pandas DataFrame.
    """
    try:
        stock = yf.Ticker(ticker)
        # Get last two days to compute price movement.
        data = stock.history(period="2d")
        if data.empty:
            logging.error("No market data returned for ticker %s", ticker)
            return None
        return data
    except Exception as e:
        logging.error("Error fetching market data: %s", e)
        return None

def calculate_technical_features(data):
    """
    Computes technical features from OHLCV data.
    For simplicity, this function extracts:
      - Today's open, high, low, close, volume (5 features)
      - And pads with zeros to reach 10 features.
    In practice, compute your full indicator set.
    """
    try:
        latest = data.iloc[-1]
        features = np.array([latest['Open'], latest['High'], latest['Low'], latest['Close'], latest['Volume']])
        # Pad with zeros.
        if features.shape[0] < 10:
            pad = np.zeros(10 - features.shape[0])
            features = np.concatenate([features, pad])
        return features.astype(np.float32)
    except Exception as e:
        logging.error("Error calculating technical features: %s", e)
        return np.zeros(10, dtype=np.float32)

def fetch_news(ticker):
    """
    Fetches the latest news headlines for the ticker.
    Here we simply mock this function.
    """
    # In practice, integrate with a news API.
    headlines = ["Company X reports record earnings", "Market volatility rises amid uncertainty"]
    return headlines

def run_finetbert(headlines):
    """
    Runs FinBERT on the given headlines and returns a sentiment embedding.
    For now, we simulate by returning a random vector.
    """
    # Replace with actual FinBERT inference.
    embedding = np.random.randn(768).astype(np.float32)
    return embedding

def compute_price_movement(data):
    """
    Computes price movement as the percentage change from the previous close to the latest close.
    """
    if data.shape[0] < 2:
        return 0.0
    prev_close = data['Close'].iloc[-2]
    last_close = data['Close'].iloc[-1]
    movement = (last_close - prev_close) / prev_close
    return float(movement)

def compute_sentiment_score(headlines):
    """
    Computes a scalar sentiment score from the headlines.
    For now, we simulate by returning a random score between -1 and 1.
    """
    return float(np.random.uniform(-1, 1))

def place_trade(ticker, signal, contradiction_type):
    """
    Places a trade (or simulates one) based on the signal.
    For this example, a positive signal triggers a BUY, negative triggers a SELL.
    Implements fixed position sizing, stop-loss, and take-profit.
    """
    action = "BUY" if signal > 0 else "SELL"
    # Log trade details. In practice, integrate with IBKR API.
    trade_details = {
        "ticker": ticker,
        "action": action,
        "position_size": POSITION_SIZE,
        "stop_loss": STOP_LOSS,
        "take_profit": TAKE_PROFIT,
        "contradiction": contradiction_type,
        "signal": signal,
        "timestamp": datetime.utcnow().isoformat()
    }
    logging.info("Placing trade: %s", trade_details)
    # Simulate trade placement.
    return trade_details

# -------------------------
# MAIN LIVE TRADING FUNCTION
# -------------------------
def run_live_trading():
    # Load the trained model and contradiction engine.
    model = FusionNet(input_dim=10, hidden_dim=512, use_attention=True, fusion_method='concat', target_mode=TARGET_MODE).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    contradiction_engine = ContradictionEngine(embedding_dim=768).to(device)
    
    # For normalized regression mode, load the target scaler (if needed for logging predictions)
    if TARGET_MODE == "normalized":
        scaler = joblib.load(SCALER_PATH)
    
    logging.info("Starting live trading for ticker %s", TICKER)
    
    # Fetch market data.
    market_data = fetch_market_data(TICKER)
    if market_data is None:
        logging.error("Failed to fetch market data.")
        return
    
    # Compute technical features.
    technical_features = calculate_technical_features(market_data)
    logging.info("Technical features: %s", technical_features)
    
    # Compute price movement.
    price_movement = compute_price_movement(market_data)
    logging.info("Price movement: %.4f", price_movement)
    
    # Fetch news headlines.
    headlines = fetch_news(TICKER)
    logging.info("Fetched headlines: %s", headlines)
    
    # Run FinBERT to get sentiment embedding.
    finbert_embedding = run_finetbert(headlines)
    logging.info("FinBERT embedding sample (first 5): %s", finbert_embedding[:5])
    
    # Compute a scalar sentiment score.
    sentiment_score = compute_sentiment_score(headlines)
    logging.info("Sentiment score: %.4f", sentiment_score)
    
    # Convert inputs to torch tensors.
    tech_tensor = torch.tensor(technical_features.reshape(1, -1), dtype=torch.float32).to(device)
    finbert_tensor = torch.tensor(finbert_embedding.reshape(1, -1), dtype=torch.float32).to(device)
    price_tensor = torch.tensor(np.array([price_movement]), dtype=torch.float32).to(device)
    sentiment_tensor = torch.tensor(np.array([sentiment_score]), dtype=torch.float32).to(device)
    
    # Run Contradiction Engine.
    updated_embedding, contradiction_type = contradiction_engine(finbert_tensor.squeeze(0), tech_tensor.squeeze(0), price_tensor.squeeze(0), sentiment_tensor.squeeze(0))
    logging.info("Contradiction type: %s", contradiction_type if contradiction_type is not None else "none")
    
    # Get prediction from model.
    with torch.no_grad():
        prediction = model(tech_tensor, updated_embedding.unsqueeze(0)).view(-1)
    prediction_value = prediction.item()
    
    # For normalized regression, inverse-transform for logging.
    if TARGET_MODE != "binary":
        prediction_logged = scaler.inverse_transform(np.array([[prediction_value]])).item()
    else:
        prediction_logged = prediction_value  # For binary, prediction is probability.
    
    logging.info("Predicted return: %.4f", prediction_logged)
    
    # Place a trade based on prediction.
    trade = place_trade(TICKER, prediction_logged, contradiction_type)
    
    # Log outcome (here we simply log the trade; integration with broker would capture execution outcome).
    logging.info("Trade executed: %s", trade)
    
    # Return trade details for further processing if needed.
    return trade

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live trading script for contradiction-aware FusionNet model.")
    parser.add_argument("--loop", action="store_true", help="Run in a loop (daily).")
    args = parser.parse_args()
    
    if args.loop:
        while True:
            run_live_trading()
            logging.info("Sleeping until next trading day...")
            time.sleep(60 * 60 * 24)  # Sleep for 24 hours.
    else:
        run_live_trading()#!/usr/bin/env python3
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
    main()import math

def black_scholes_price(S, K, T, r, sigma, option_type="call"):
    """
    Calculates the Black-Scholes price for European options.
    
    Args:
        S (float): Current stock price.
        K (float): Strike price.
        T (float): Time to expiration (in years).
        r (float): Risk-free rate (annualized).
        sigma (float): Implied volatility (annualized).
        option_type (str): "call" or "put".
        
    Returns:
        price (float): Option price.
    """
    if T <= 0 or sigma <= 0:
        return max(0.0, S - K) if option_type == "call" else max(0.0, K - S)
    
    d1 = (math.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    
    from scipy.stats import norm
    if option_type == "call":
        price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    return priceimport numpy as np
import torch
import joblib
import argparse
from fusionnet import FusionNet
from contradiction_engine import ContradictionEngine

parser = argparse.ArgumentParser()
parser.add_argument("--target_mode", type=str, default="normalized", choices=["normalized", "binary", "rolling"])
parser.add_argument("--filter_underhype", action="store_true", help="Filter predictions to only include 'underhype' samples.")
args = parser.parse_args()
target_mode = args.target_mode
filter_underhype = args.filter_underhype

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = FusionNet(input_dim=10, hidden_dim=512, use_attention=True, fusion_method='concat', target_mode=target_mode).to(device)
model.load_state_dict(torch.load("./training_data/fusion_underhype_weights_fold5.pth", map_location=device))
model.eval()
contradiction_engine = ContradictionEngine(embedding_dim=768).to(device)

data = np.load("./training_data/dataset.npz")
tech_data = data["technical_features"]
finbert_data = data["finbert_embeddings"]
price_data = data["price_movements"]
sentiment_data = data["news_sentiment_scores"]

tech_tensor = torch.tensor(tech_data, dtype=torch.float32).to(device)
finbert_tensor = torch.tensor(finbert_data, dtype=torch.float32).to(device)
price_tensor = torch.tensor(price_data, dtype=torch.float32).to(device)
sentiment_tensor = torch.tensor(sentiment_data, dtype=torch.float32).to(device)

updated_embeddings = []
contradiction_tags = []
for i in range(finbert_tensor.size(0)):
    updated_emb, ctype = contradiction_engine(finbert_tensor[i], tech_tensor[i], price_tensor[i], sentiment_tensor[i])
    updated_embeddings.append(updated_emb)
    contradiction_tags.append(ctype if ctype is not None else "none")
updated_embeddings = torch.stack(updated_embeddings)

with torch.no_grad():
    predictions = model(tech_tensor, updated_embeddings)
predictions = predictions.cpu().numpy().flatten()

if target_mode != "binary":
    scaler = joblib.load("./training_data/target_scaler.pkl")
    predictions_final = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
else:
    predictions_final = predictions

if filter_underhype:
    # Zero out predictions for samples not labeled "underhype"
    tags = np.array(contradiction_tags)
    mask = tags == "underhype"
    num_filtered = np.sum(~mask)
    predictions_final[~mask] = 0.0
    print(f"Filtering applied: {num_filtered} samples zeroed out (not underhype).")

print("Predictions stats:")
print(f"  Before transform: min {predictions.min()}, max {predictions.max()}, mean {predictions.mean()}")
print(f"  Final predictions: min {predictions_final.min()}, max {predictions_final.max()}, mean {predictions_final.mean()}")

np.savez("./training_data/predictions.npz",
         predictions=predictions_final,
         target_returns=data["target_returns"],
         contradiction_tags=np.array(contradiction_tags))
print("Predictions saved to ./training_data/predictions.npz")
import numpy as np
import torch
import argparse
import joblib
from fusionnet import FusionNet
from contradiction_engine import ContradictionEngine

def parse_args():
    parser = argparse.ArgumentParser(description="Route test samples to contradiction-specific FusionNet models.")
    parser.add_argument("--target_mode", type=str, default="normalized", choices=["normalized", "binary", "rolling"], help="Target mode used in training.")
    parser.add_argument("--contradiction_filter", type=str, default="all", choices=["underhype", "overhype", "none", "all"], help="Filter predictions by a specific contradiction type.")
    return parser.parse_args()

def main():
    args = parse_args()
    target_mode = args.target_mode
    filter_type = args.contradiction_filter

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load the full dataset.
    data = np.load("./training_data/dataset.npz")
    tech_data = data["technical_features"]         # Shape: [N, d_tech]
    finbert_data = data["finbert_embeddings"]        # Shape: [N, 768]
    price_data = data["price_movements"]             # Shape: [N]
    sentiment_data = data["news_sentiment_scores"]     # Shape: [N]
    target_returns = data["target_returns"]          # Shape: [N, 1]
    target_returns = target_returns.flatten()        # Make it 1D.

    # Convert dataset arrays to torch tensors.
    tech_tensor = torch.tensor(tech_data, dtype=torch.float32).to(device)
    finbert_tensor = torch.tensor(finbert_data, dtype=torch.float32).to(device)
    price_tensor = torch.tensor(price_data, dtype=torch.float32).to(device)
    sentiment_tensor = torch.tensor(sentiment_data, dtype=torch.float32).to(device)

    N = tech_tensor.size(0)
    print(f"Loaded {N} samples from dataset.")

    # Load all three models.
    # Assumes FusionNet was trained with fusion_method 'concat' and target_mode as specified.
    model_underhype = FusionNet(input_dim=tech_tensor.shape[1], hidden_dim=512, use_attention=True,
                                fusion_method='concat', target_mode=target_mode).to(device)
    model_overhype = FusionNet(input_dim=tech_tensor.shape[1], hidden_dim=512, use_attention=True,
                               fusion_method='concat', target_mode=target_mode).to(device)
    model_none = FusionNet(input_dim=tech_tensor.shape[1], hidden_dim=512, use_attention=True,
                           fusion_method='concat', target_mode=target_mode).to(device)
    model_underhype.load_state_dict(torch.load("./training_data/fusion_underhype_weights.pth", map_location=device))
    model_overhype.load_state_dict(torch.load("./training_data/fusion_overhype_weights.pth", map_location=device))
    model_none.load_state_dict(torch.load("./training_data/fusion_none_weights.pth", map_location=device))
    model_underhype.eval()
    model_overhype.eval()
    model_none.eval()
    print("All specialized models loaded.")

    # Load the ContradictionEngine (its role here is to determine the tag only).
    contr_engine = ContradictionEngine(embedding_dim=768).to(device)
    contr_engine.eval()

    # Prepare lists for predictions and tags.
    predictions = []
    contradiction_tags = []

    # Loop over all samples.
    with torch.no_grad():
        for i in range(N):
            tech_sample = tech_tensor[i]             # Shape: [d_tech]
            finbert_sample = finbert_tensor[i]         # Shape: [768]
            price_sample = price_tensor[i]             # Scalar tensor.
            sentiment_sample = sentiment_tensor[i]     # Scalar tensor.
            
            # Determine contradiction tag using ContradictionEngine.
            # We ignore the updated embedding since each head expects raw FinBERT.
            _, ctype = contr_engine(finbert_sample, tech_sample, price_sample, sentiment_sample)
            if ctype is None:
                ctype = "none"
            contradiction_tags.append(ctype)
            
            # Route the sample based on its contradiction tag.
            # Each head expects the raw FinBERT embedding.
            tech_input = tech_sample.unsqueeze(0)       # Shape: [1, d_tech]
            finbert_input = finbert_sample.unsqueeze(0)   # Shape: [1, 768]
            if ctype == "underhype":
                pred = model_underhype(tech_input, finbert_input)
            elif ctype == "overhype":
                pred = model_overhype(tech_input, finbert_input)
            else:
                pred = model_none(tech_input, finbert_input)
            predictions.append(pred.cpu().detach().numpy().flatten()[0])
    
    predictions = np.array(predictions)
    contradiction_tags = np.array(contradiction_tags)

    # If filtering is requested, zero out predictions for samples not matching the filter.
    if filter_type != "all":
        mask = (contradiction_tags == filter_type)
        num_filtered = np.sum(~mask)
        predictions[~mask] = 0.0
        print(f"Filtering applied: {num_filtered} samples zeroed out (not '{filter_type}').")
    
    # Save final predictions.
    output_path = "./training_data/predictions_routed.npz"
    np.savez(output_path,
             predictions=predictions,
             target_returns=target_returns,
             contradiction_tags=contradiction_tags)
    print("Predictions saved to", output_path)

if __name__ == "__main__":
    args = parse_args()
    main()
# prepare_dataset.py
import numpy as np
import pandas as pd
import os
import argparse
from sklearn.preprocessing import StandardScaler
import joblib

def prepare_dataset(csv_path, output_npz_path, scaler_path, target_mode="normalized"):
    df = pd.read_csv(csv_path)
    
    # Extract features
    technical_cols = [f"tech{i}" for i in range(1, 11)]
    finbert_cols = [f"finbert_{i}" for i in range(768)]
    price_col = "price_movement"
    sentiment_col = "news_sentiment_score"

    required_cols = technical_cols + finbert_cols + [price_col, sentiment_col, "next_return"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    # Filter valid rows
    df = df.dropna(subset=required_cols)

    # Scale technical features
    scaler = StandardScaler()
    tech_scaled = scaler.fit_transform(df[technical_cols])
    joblib.dump(scaler, scaler_path)
    print("✅ Technical features normalized.")

    # Prepare arrays
    finbert_embeddings = df[finbert_cols].values.astype(np.float32)
    price_movements = df[price_col].values.astype(np.float32)
    sentiment_scores = df[sentiment_col].values.astype(np.float32)

    if target_mode == "binary":
        target_returns = (df["next_return"] > 0).astype(np.float32)
    else:
        target_returns = df["next_return"].values.astype(np.float32)
        if target_mode == "normalized":
            target_scaler = StandardScaler()
            target_returns = target_scaler.fit_transform(target_returns.reshape(-1, 1)).flatten()
            joblib.dump(target_scaler, "./training_data/target_scaler.pkl")

    # Save as .npz
    np.savez(output_npz_path,
             technical_features=tech_scaled,
             finbert_embeddings=finbert_embeddings,
             price_movements=price_movements,
             news_sentiment_scores=sentiment_scores,
             target_returns=target_returns)
    
    print(f"✅ Dataset saved to {output_npz_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_mode", type=str, default="normalized", choices=["normalized", "binary", "rolling"])
    args = parser.parse_args()

    csv_path = "./training_data/combined_data_with_target_with_real_finbert.csv"
    output_npz_path = "./training_data/dataset.npz"
    scaler_path = "./training_data/technical_scaler.pkl"

    prepare_dataset(csv_path, output_npz_path, scaler_path, target_mode=args.target_mode)import argparse
import os
import subprocess

def parse_args():
    parser = argparse.ArgumentParser(description="Run full contradiction-aware trading pipeline.")
    parser.add_argument("--csv", type=str, help="Path to CSV data file.")
    parser.add_argument("--model_path", type=str, default="./training_data/fusion_net_contradiction_weights.pth", help="Path to save/load the model.")
    parser.add_argument("--mode", type=str, choices=["prepare", "train", "predict", "evaluate"], default="train", help="Pipeline mode.")
    parser.add_argument("--target_mode", type=str, default="normalized", choices=["normalized", "binary", "rolling"], help="Target mode for training/prediction.")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs("./training_data", exist_ok=True)
    
    if args.mode == "prepare":
        subprocess.run(["python", "prepare_dataset.py", "--csv", args.csv, "--target_mode", args.target_mode])
    elif args.mode == "train":
        if not os.path.exists("./training_data/dataset.npz"):
            subprocess.run(["python", "prepare_dataset.py", "--csv", args.csv, "--target_mode", args.target_mode])
        subprocess.run(["python", "train_fusion.py", "--target_mode", args.target_mode])
    elif args.mode == "predict":
        subprocess.run(["python", "predict_fusion.py", "--target_mode", args.target_mode])
    elif args.mode == "evaluate":
        subprocess.run(["python", "evaluate_strategy.py", "--target_mode", args.target_mode])
    else:
        print("Unsupported mode.")

if __name__ == "__main__":
    main()# temporal_encoder.py
import torch
import torch.nn as nn

class TransformerTimeSeriesEncoder(nn.Module):
    def __init__(self, input_dim, model_dim, num_layers=2, nhead=4, dropout=0.1):
        super(TransformerTimeSeriesEncoder, self).__init__()
        self.input_fc = nn.Linear(input_dim, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_fc = nn.Linear(model_dim, model_dim)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch, seq_len, input_dim]
        Returns:
            encoded: Tensor of shape [batch, model_dim] (e.g. mean-pooled over sequence)
        """
        x = self.input_fc(x)
        x = self.transformer_encoder(x)
        x = self.output_fc(x)
        # Mean pool over sequence dimension.
        encoded = x.mean(dim=1)
        return encoded

if __name__ == "__main__":
    # Quick test.
    batch_size = 8
    seq_len = 10
    input_dim = 20
    model_dim = 32
    dummy_input = torch.randn(batch_size, seq_len, input_dim)
    encoder = TransformerTimeSeriesEncoder(input_dim, model_dim)
    output = encoder(dummy_input)
    print("Transformer encoder output shape:", output.shape)#!/usr/bin/env python3
"""
train_contradiction_heads.py

Trains a separate FusionNet head on a filtered dataset corresponding to a specific contradiction type.
Usage:
    python train_contradiction_heads.py --dataset_path ./training_data/underhype_only_dataset.npz --contradiction_type underhype --target_mode normalized --epochs 50 --batch_size 128
"""
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import joblib
from fusionnet import FusionNet

# Set device.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def compute_direction_accuracy(predictions, targets):
    pred_dir = predictions > 0
    target_dir = targets > 0
    return np.mean(pred_dir == target_dir)

def train_model(dataset_path, contradiction_type, target_mode, epochs, batch_size):
    # Load filtered dataset.
    data = np.load(dataset_path)
    tech_data = data["technical_features"]         # shape: [N, 10(+optional)]
    finbert_data = data["finbert_embeddings"]        # shape: [N, 768]
    price_data = data["price_movements"]             # shape: [N]
    sentiment_data = data["news_sentiment_scores"]     # shape: [N]
    target_returns = data["target_returns"]          # shape: [N,1]
    
    print(f"Training on dataset for contradiction type: {contradiction_type}")
    print("Total samples:", tech_data.shape[0])
    
    # Normalize technical features.
    tech_scaler = StandardScaler()
    tech_data_scaled = tech_scaler.fit_transform(tech_data)
    
    # Optionally, save the technical scaler.
    joblib.dump(tech_scaler, f"./training_data/tech_scaler_{contradiction_type}.pkl")
    print(f"Technical scaler saved to ./training_data/tech_scaler_{contradiction_type}.pkl")
    
    # Convert to torch tensors.
    tech_tensor = torch.tensor(tech_data_scaled, dtype=torch.float32).to(device)
    finbert_tensor = torch.tensor(finbert_data, dtype=torch.float32).to(device)
    price_tensor = torch.tensor(price_data, dtype=torch.float32).to(device)
    sentiment_tensor = torch.tensor(sentiment_data, dtype=torch.float32).to(device)
    target_tensor = torch.tensor(target_returns, dtype=torch.float32).to(device)
    
    # Instantiate model.
    model = FusionNet(input_dim=tech_tensor.shape[1], hidden_dim=512, use_attention=True, fusion_method='concat', target_mode=target_mode).to(device)
    
    # Choose loss.
    if target_mode == "binary":
        loss_fn = nn.BCEWithLogitsLoss()
    else:
        loss_fn = nn.MSELoss()
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    num_samples = tech_tensor.shape[0]
    num_batches = (num_samples + batch_size - 1) // batch_size
    loss_history = []
    
    model.train()
    for epoch in range(epochs):
        permutation = torch.randperm(num_samples)
        epoch_loss = 0.0
        for i in range(num_batches):
            indices = permutation[i * batch_size: (i+1) * batch_size]
            batch_tech = tech_tensor[indices]
            batch_finbert = finbert_tensor[indices]
            batch_price = price_tensor[indices]
            batch_sentiment = sentiment_tensor[indices]
            batch_target = target_tensor[indices]
            
            optimizer.zero_grad()
            # For training on filtered dataset, we assume samples already match the contradiction type.
            # So we simply use the original FinBERT embeddings.
            preds = model(batch_tech, batch_finbert).view(-1)
            loss = loss_fn(preds, batch_target.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item() * batch_tech.size(0)
        avg_loss = epoch_loss / num_samples
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    # Save the trained model.
    model_save_path = f"./training_data/fusion_{contradiction_type}_weights.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Training complete. Model saved to {model_save_path}")
    
    return model, loss_history

def main():
    parser = argparse.ArgumentParser(description="Train a FusionNet head for a specific contradiction type.")
    parser.add_argument("--dataset_path", type=str, default="./training_data/underhype_only_dataset.npz", help="Path to filtered dataset .npz file.")
    parser.add_argument("--contradiction_type", type=str, default="underhype", choices=["underhype", "overhype", "none"], help="Contradiction type to train on.")
    parser.add_argument("--target_mode", type=str, default="normalized", choices=["normalized", "binary", "rolling"], help="Target mode.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")
    args = parser.parse_args()
    
    model, loss_history = train_model(args.dataset_path, args.contradiction_type, args.target_mode, args.epochs, args.batch_size)
    
if __name__ == "__main__":
    main()import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import RobustScaler  # or StandardScaler, as needed
import matplotlib.pyplot as plt

# Import your new model components from your repo.
# Assuming your contradiction-aware model is defined in jackpot/model/contradiction_model.py
from contradiction_model import TradingModel, ContradictionLoss

# ================================
# 1. Data Preparation & Normalization
# ================================
# Load actual data
df = pd.read_csv("training_data/combined_data_with_target.csv")

# Define the technical features used in add_target.py
technical_cols = ['SMA20', 'SMA50', 'EMA20', 'EMA50', 'RSI14', 'MACD', 
                  'StochK', 'StochD', 'HistoricalVol20', 'ATR14']

# Extract inputs and targets
technical_features = df[technical_cols].values
target_returns = df["next_return"].values.reshape(-1, 1)

# TEMPORARY: Use random embeddings until FinBERT is ready
finbert_embeddings = np.random.rand(len(df), 768)

# Normalize technicals.
# Using RobustScaler as it is robust to outliers.
scaler = RobustScaler()
technical_features_scaled = scaler.fit_transform(technical_features)

# Optionally, if ATR or volume features need log transformation, apply here.
# For example, if column indices 7 and 8 are ATR/volume:
# technical_features_scaled[:, 7] = np.log1p(technical_features_scaled[:, 7])
# technical_features_scaled[:, 8] = np.log1p(technical_features_scaled[:, 8])

# ================================
# 2. Model Setup
# ================================
# Hyperparameters
tech_input_dim = 10
sentiment_input_dim = 768
encoder_hidden_dim = 64
proj_dim = 32
decision_hidden_dim = 64

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate the model and move to device.
model = TradingModel(
    tech_input_dim=tech_input_dim,
    sentiment_input_dim=sentiment_input_dim,
    encoder_hidden_dim=encoder_hidden_dim,
    proj_dim=proj_dim,
    decision_hidden_dim=decision_hidden_dim
)
model.to(device)

# Initialize loss functions.
primary_loss_fn = nn.MSELoss()
contradiction_loss_fn = ContradictionLoss(weight=0.5)

# Setup optimizer.
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Convert numpy arrays to torch tensors.
tech_tensor = torch.tensor(technical_features_scaled, dtype=torch.float32).to(device)
sentiment_tensor = torch.tensor(finbert_embeddings, dtype=torch.float32).to(device)
target_tensor = torch.tensor(target_returns, dtype=torch.float32).to(device)

# ================================
# 3. Training Loop
# ================================
num_epochs = 20
batch_size = 32
num_samples = tech_tensor.shape[0]
num_batches = int(np.ceil(num_samples / batch_size))

for epoch in range(num_epochs):
    model.train()
    permutation = torch.randperm(num_samples)
    epoch_loss = 0.0
    for i in range(num_batches):
        indices = permutation[i * batch_size : (i + 1) * batch_size]
        batch_tech = tech_tensor[indices]
        batch_sent = sentiment_tensor[indices]
        batch_target = target_tensor[indices]
        
        optimizer.zero_grad()
        # Forward pass through the model.
        decision, contradiction_score, proj_tech, proj_sent, gate_weight = model(batch_tech, batch_sent)
        
        # Compute the primary prediction loss.
        primary_loss = primary_loss_fn(decision.view(-1, 1), batch_target)
        # Compute the auxiliary contradiction loss.
        contr_loss = contradiction_loss_fn(proj_tech, proj_sent, decision)
        
        loss = primary_loss + contr_loss
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / num_batches
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

# ================================
# 4. Saving & Loading the Model
# ================================
model_path = "contradiction_model.pth"
torch.save(model.state_dict(), model_path)
print("Model saved at:", model_path)

# To load the model later:
loaded_model = TradingModel(tech_input_dim, sentiment_input_dim, encoder_hidden_dim, proj_dim, decision_hidden_dim)
loaded_model.load_state_dict(torch.load(model_path))
loaded_model.to(device)
loaded_model.eval()

# ================================
# 5. Quick Testing (Forward Pass)
# ================================
# Test with a small slice (e.g., first 5 samples).
model.eval()
with torch.no_grad():
    test_tech = tech_tensor[:5]
    test_sent = sentiment_tensor[:5]
    test_decision, test_contradiction_score, test_proj_tech, test_proj_sent, test_gate_weight = model(test_tech, test_sent)
    print("Test Decision Output:", test_decision)
    print("Test Contradiction Scores:", test_contradiction_score)
    print("Test Gate Weights:", test_gate_weight)

# ================================
# 6. (Optional) Bonus: Visualization & Transformer Encoder Idea
# ================================
# To visualize contradiction scores and gate weights:
with torch.no_grad():
    full_decision, full_contradiction_score, full_proj_tech, full_proj_sent, full_gate_weight = model(tech_tensor, sentiment_tensor)
    # Convert to numpy for plotting.
    contradiction_scores_np = full_contradiction_score.cpu().numpy()
    gate_weights_np = full_gate_weight.cpu().numpy().flatten()
    
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(contradiction_scores_np, bins=30)
plt.title("Histogram of Contradiction Scores")
plt.subplot(1, 2, 2)
plt.hist(gate_weights_np, bins=30)
plt.title("Histogram of Gate Weights")
plt.show()

# BONUS: Adding a Transformer-based encoder for time series.
# Later, you might want to replace or supplement your EncoderTechnical with a Transformer encoder.
# For example:
#
# from torch.nn import TransformerEncoder, TransformerEncoderLayer
#
# class TransformerTimeSeriesEncoder(nn.Module):
#     def __init__(self, input_dim, model_dim, num_layers, nhead, dropout=0.1):
#         super(TransformerTimeSeriesEncoder, self).__init__()
#         self.input_fc = nn.Linear(input_dim, model_dim)
#         encoder_layer = TransformerEncoderLayer(d_model=model_dim, nhead=nhead, dropout=dropout)
#         self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)
#         self.output_fc = nn.Linear(model_dim, model_dim)
#     def forward(self, x):
#         # x should be of shape (batch, seq_len, input_dim)
#         x = self.input_fc(x)
#         x = self.transformer_encoder(x)
#         x = self.output_fc(x)
#         # Example: take the mean across the sequence dimension.
#         return x.mean(dim=1)
#
# You could integrate this TransformerTimeSeriesEncoder into your overall model for time-series modeling.import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from fusionnet import FusionNet
from contradiction_engine import ContradictionEngine
from sklearn.preprocessing import StandardScaler

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Args
parser = argparse.ArgumentParser()
parser.add_argument("--target_mode", type=str, default="normalized", choices=["normalized", "binary", "rolling"])
parser.add_argument("--epochs", type=int, default=75, help="Number of training epochs")
args = parser.parse_args()
target_mode = args.target_mode
num_epochs = args.epochs
print(f"Training with target_mode: {target_mode}, epochs: {num_epochs}")

# Load data
data = np.load("./training_data/dataset.npz")
tech_data = data["technical_features"]
finbert_data = data["finbert_embeddings"]
price_data = data["price_movements"]
sentiment_data = data["news_sentiment_scores"]
target_returns = data["target_returns"]

# Normalize technical features (safe)
scaler_tech = StandardScaler()
tech_data_scaled = scaler_tech.fit_transform(tech_data)
tech_data_scaled = np.nan_to_num(tech_data_scaled, nan=0.0, posinf=0.0, neginf=0.0)

# Clamp embeddings to avoid exploding values
finbert_data = np.clip(finbert_data, -10.0, 10.0)

# Convert to tensors
tech_tensor = torch.tensor(tech_data_scaled, dtype=torch.float32).to(device)
finbert_tensor = torch.tensor(finbert_data, dtype=torch.float32).to(device)
price_tensor = torch.tensor(price_data, dtype=torch.float32).to(device)
sentiment_tensor = torch.tensor(sentiment_data, dtype=torch.float32).to(device)
target_tensor = torch.tensor(target_returns, dtype=torch.float32).to(device)

# Model and optimizer
model = FusionNet(
    input_dim=tech_tensor.shape[1],
    hidden_dim=512,
    use_attention=True,
    fusion_method='concat',
    target_mode=target_mode
).to(device)
contradiction_engine = ContradictionEngine(embedding_dim=768).to(device)

loss_fn = nn.BCEWithLogitsLoss() if target_mode == "binary" else nn.MSELoss()
optimizer = optim.Adam(list(model.parameters()) + list(contradiction_engine.parameters()), lr=1e-4)

# Training loop
batch_size = 128
num_samples = tech_tensor.shape[0]
num_batches = (num_samples + batch_size - 1) // batch_size
loss_history = []

model.train()
for epoch in range(num_epochs):
    epoch_loss = 0.0
    permutation = torch.randperm(num_samples)

    for i in range(num_batches):
        indices = permutation[i * batch_size : (i + 1) * batch_size]
        batch_tech = tech_tensor[indices]
        batch_finbert = finbert_tensor[indices]
        batch_price = price_tensor[indices]
        batch_sentiment = sentiment_tensor[indices]
        batch_target = target_tensor[indices]

        # Safety checks
        if torch.isnan(batch_tech).any() or torch.isinf(batch_tech).any():
            print("🚨 NaN or Inf in tech features. Skipping batch.")
            continue

        if torch.isnan(batch_finbert).any() or torch.isinf(batch_finbert).any():
            print("🚨 NaN or Inf in FinBERT embeddings. Skipping batch.")
            continue

        optimizer.zero_grad()
        updated_embeddings = []
        for j in range(batch_finbert.size(0)):
            updated_emb, _ = contradiction_engine(
                batch_finbert[j], batch_tech[j], batch_price[j], batch_sentiment[j]
            )
            updated_embeddings.append(updated_emb)

        updated_embeddings = torch.stack(updated_embeddings)
        prediction = model(batch_tech, updated_embeddings).view(-1)
        loss = loss_fn(prediction, batch_target.view(-1))

        # Sanity check
        if torch.isnan(loss) or torch.isinf(loss):
            print("🚨 NaN/Inf loss detected. Skipping batch.")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_loss += loss.item() * batch_finbert.size(0)

    avg_loss = epoch_loss / num_samples
    loss_history.append(avg_loss)
    print(f"✅ Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")

# Save model
torch.save(model.state_dict(), "./training_data/fusion_net_contradiction_weights.pth")
print("✅ Training complete. Model saved to ./training_data/fusion_net_contradiction_weights.pth")

# Optional: plot loss curve
try:
    import matplotlib.pyplot as plt
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.savefig("./training_data/loss_curve.png")
    print("📈 Loss curve saved to training_data/loss_curve.png")
except ImportError:
    print("matplotlib not installed. Skipping loss plot.")#!/usr/bin/env python3
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from fusionnet import FusionNet
from contradiction_engine import ContradictionEngine
# (Assume other modules like StandardScaler are imported as needed)

def compute_direction_accuracy(predictions, targets):
    pred_dir = predictions > 0
    target_dir = targets > 0
    return np.mean(pred_dir == target_dir)

def compute_sharpe_ratio(returns, risk_free_rate=0.0):
    excess = returns - risk_free_rate
    return excess.mean() / (excess.std() + 1e-8)

def run_kfold_cv(dataset_path, n_splits, num_epochs, batch_size, target_mode, save_models=False):
    data = np.load(dataset_path)
    tech_data = data["technical_features"]
    finbert_data = data["finbert_embeddings"]
    price_data = data["price_movements"]
    sentiment_data = data["news_sentiment_scores"]
    target_returns = data["target_returns"]
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    metrics = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    fold = 1
    for train_idx, val_idx in kf.split(tech_data):
        print(f"Starting Fold {fold}")
        # Prepare training tensors.
        tech_train = torch.tensor(tech_data[train_idx], dtype=torch.float32).to(device)
        finbert_train = torch.tensor(finbert_data[train_idx], dtype=torch.float32).to(device)
        price_train = torch.tensor(price_data[train_idx], dtype=torch.float32).to(device)
        sentiment_train = torch.tensor(sentiment_data[train_idx], dtype=torch.float32).to(device)
        target_train = torch.tensor(target_returns[train_idx], dtype=torch.float32).to(device)
        
        tech_val = torch.tensor(tech_data[val_idx], dtype=torch.float32).to(device)
        finbert_val = torch.tensor(finbert_data[val_idx], dtype=torch.float32).to(device)
        price_val = torch.tensor(price_data[val_idx], dtype=torch.float32).to(device)
        sentiment_val = torch.tensor(sentiment_data[val_idx], dtype=torch.float32).to(device)
        target_val = torch.tensor(target_returns[val_idx], dtype=torch.float32).to(device)
        
        model = FusionNet(input_dim=tech_train.shape[1], hidden_dim=512, use_attention=True, fusion_method='concat', target_mode=target_mode).to(device)
        contr_engine = ContradictionEngine(embedding_dim=768).to(device)
        if target_mode == "binary":
            loss_fn = nn.BCEWithLogitsLoss()
        else:
            loss_fn = nn.MSELoss()
        optimizer = optim.Adam(list(model.parameters()) + list(contr_engine.parameters()), lr=1e-3)
        
        n_train = tech_train.shape[0]
        num_batches = (n_train + batch_size - 1) // batch_size
        
        for epoch in range(num_epochs):
            model.train()
            contr_engine.train()
            permutation = torch.randperm(n_train)
            epoch_loss = 0.0
            for i in range(num_batches):
                indices = permutation[i*batch_size:(i+1)*batch_size]
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
            print(f"[Fold {fold}] Epoch {epoch+1}/{num_epochs} Loss: {avg_loss:.4f}")
        
        model.eval()
        contr_engine.eval()
        with torch.no_grad():
            updated_val_embeddings = []
            for j in range(finbert_val.size(0)):
                upd_emb, _ = contr_engine(finbert_val[j], tech_val[j], price_val[j], sentiment_val[j])
                updated_val_embeddings.append(upd_emb)
            updated_val_embeddings = torch.stack(updated_val_embeddings)
            val_preds = model(tech_val, updated_val_embeddings).view(-1).cpu().numpy()
            val_targets = target_val.view(-1).cpu().numpy()
        dir_acc = compute_direction_accuracy(val_preds, val_targets)
        avg_ret = val_preds.mean()
        sharpe = compute_sharpe_ratio(val_preds)
        metrics.append({"direction_accuracy": dir_acc, "average_return": avg_ret, "sharpe_ratio": sharpe})
        print(f"[Fold {fold}] Metrics: Direction Accuracy: {dir_acc:.2%}, Avg Return: {avg_ret:.4f}, Sharpe: {sharpe:.4f}")
        # Optionally save the model.
        if save_models:
            torch.save(model.state_dict(), f"./training_data/fusion_underhype_weights_fold{fold}.pth")
            print(f"Model for fold {fold} saved.")
        fold += 1
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Train contradiction-aware model with 5-fold CV.")
    parser.add_argument("--dataset_path", type=str, default="./training_data/dataset.npz", help="Path to dataset .npz file.")
    parser.add_argument("--target_mode", type=str, default="normalized", choices=["normalized", "binary", "rolling"])
    parser.add_argument("--epochs", type=int, default=75, help="Epochs per fold.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")
    parser.add_argument("--n_splits", type=int, default=5, help="Number of folds.")
    parser.add_argument("--save_models", action="store_true", help="Save model for each fold.")
    args = parser.parse_args()
    
    metrics = run_kfold_cv(args.dataset_path, args.n_splits, args.epochs, args.batch_size, args.target_mode, args.save_models)
    avg_dir = np.mean([m["direction_accuracy"] for m in metrics])
    avg_ret = np.mean([m["average_return"] for m in metrics])
    avg_sharpe = np.mean([m["sharpe_ratio"] for m in metrics])
    print("Average Metrics Across Folds:")
    print(f"  Direction Accuracy: {avg_dir:.2%}")
    print(f"  Average Return: {avg_ret:.4f}")
    print(f"  Sharpe Ratio: {avg_sharpe:.4f}")

if __name__ == "__main__":
    main()import optuna
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from fusionnet import FusionNet

# Set device for GPU acceleration.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

###############################################
# Preprocessing Block
###############################################

# Load CSV and inspect columns.
data = pd.read_csv("training_data/combined_data_with_target.csv")
print("Initial data shape:", data.shape)
print("Columns in CSV:", data.columns.tolist())

# Define feature columns.
feature_columns = [
    "Close", "Open", "High", "Low", "Volume", "SMA20", 
    "SMA50", "EMA20", "EMA50", "RSI14", "MACD", "StochK", 
    "StochD", "HistoricalVol20", "ATR14", "ImpliedVol"
]

# Replace infinities with NaN.
data.replace([np.inf, -np.inf], np.nan, inplace=True)

# Print number of NaNs per feature column.
print("Missing values per feature column before cleaning:")
print(data[feature_columns].isna().sum())

# Only drop rows where the target is missing.
data.dropna(subset=['next_return'], inplace=True)
print("Data shape after dropping rows with missing target:", data.shape)

# Fill missing feature values (using forward-fill; adjust method as needed).
data[feature_columns] = data[feature_columns].fillna(method='ffill')
print("Missing values per feature column after forward-fill:")
print(data[feature_columns].isna().sum())

# Extract features and target.
X_raw = data[feature_columns].values
y = data['next_return'].values

# Debug: Print shapes to confirm data isn't empty.
print("Shape of raw features (X_raw):", X_raw.shape)
print("Shape of target (y):", y.shape)

# Normalize features.
scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

# Confirm no NaNs are present after scaling.
if np.isnan(X).any():
    print("Warning: NaNs found in normalized features!")
if np.isnan(y).any():
    print("Warning: NaNs found in target!")

print("Shape of normalized features (X):", X.shape)

###############################################
# End of Preprocessing Block
###############################################

# Split data into training (80%) and validation (20%).
split_idx = int(len(X) * 0.8)
X_train, y_train = X[:split_idx], y[:split_idx]
X_val, y_val = X[split_idx:], y[split_idx:]

# Create an Optuna study with SQLite storage.
study = optuna.create_study(
    direction="minimize",
    storage="sqlite:///optuna_study.db",
    load_if_exists=True,
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=2)
)

def objective(trial):
    hidden_dim = trial.suggest_int('hidden_dim', 32, 128)
    use_attention = trial.suggest_categorical('use_attention', [False, True])
    fusion_method = trial.suggest_categorical('fusion_method', ['concat', 'average'])
    lr = trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True)
    
    input_dim = X_train.shape[1]
    model = FusionNet(input_dim, hidden_dim=hidden_dim, use_attention=use_attention, fusion_method=fusion_method).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    model.train()
    
    batch_size = 128
    num_samples = X_train.shape[0]
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    for epoch in range(5):
        epoch_loss = 0.0
        permutation = np.random.permutation(num_samples)
        X_train_shuffled = X_train[permutation]
        y_train_shuffled = y_train[permutation]
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i+1) * batch_size, num_samples)
            batch_X = torch.from_numpy(X_train_shuffled[start_idx:end_idx].astype(np.float32)).to(device)
            batch_y = torch.from_numpy(y_train_shuffled[start_idx:end_idx].astype(np.float32)).to(device)
            optimizer.zero_grad()
            pred = model(batch_X, batch_X)
            pred = pred.view(-1)
            loss = loss_fn(pred, batch_y)
            if torch.isnan(loss):
                print(f"NaN loss detected at epoch {epoch+1}, batch {i+1}. Pruning trial.")
                raise optuna.TrialPruned("Loss became NaN")
            with torch.autograd.set_detect_anomaly(True):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            print(f"[Trial {trial.number}] Epoch {epoch+1}, Batch {i+1}/{num_batches}, Loss: {loss.item():.4f}")
            epoch_loss += loss.item() * (end_idx - start_idx)
        epoch_loss /= num_samples
        trial.report(epoch_loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    model.eval()
    with torch.no_grad():
        Xv_tensor = torch.from_numpy(X_val.astype(np.float32)).to(device)
        yv_tensor = torch.from_numpy(y_val.astype(np.float32)).to(device)
        pred_val = model(Xv_tensor, Xv_tensor).view(-1)
        val_loss = loss_fn(pred_val, yv_tensor).item()
    return val_loss

study.optimize(objective, n_trials=50, show_progress_bar=True)
best_params = study.best_params
print("Optuna best parameters:", best_params)

# Train a final model on the combined train+val dataset.
batch_size = 128
X_all = np.concatenate([X_train, X_val], axis=0)
y_all = np.concatenate([y_train, y_val], axis=0)
num_samples_all = X_all.shape[0]
num_batches_all = (num_samples_all + batch_size - 1) // batch_size

best_model = FusionNet(input_dim, hidden_dim=best_params['hidden_dim'],
                       use_attention=best_params['use_attention'],
                       fusion_method=best_params['fusion_method']).to(device)
best_model.train()
optimizer = optim.Adam(best_model.parameters(), lr=best_params['learning_rate'])
loss_fn = nn.MSELoss()

for epoch in range(5):
    epoch_loss = 0.0
    permutation = np.random.permutation(num_samples_all)
    X_all_shuffled = X_all[permutation]
    y_all_shuffled = y_all[permutation]
    for i in range(num_batches_all):
        start_idx = i * batch_size
        end_idx = min((i+1) * batch_size, num_samples_all)
        batch_X = torch.from_numpy(X_all_shuffled[start_idx:end_idx].astype(np.float32)).to(device)
        batch_y = torch.from_numpy(y_all_shuffled[start_idx:end_idx].astype(np.float32)).to(device)
        optimizer.zero_grad()
        pred_all = best_model(batch_X, batch_X).view(-1)
        loss = loss_fn(pred_all, batch_y)
        with torch.autograd.set_detect_anomaly(True):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(best_model.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_loss += loss.item() * (end_idx - start_idx)
    epoch_loss /= num_samples_all
    print(f"Final training epoch {epoch+1}/5, Loss: {epoch_loss:.4f}")
        
best_model.eval()
best_model.save_model("fusion_net_best_weights.pth")
print("Best model saved as fusion_net_best_weights.pth")#!/usr/bin/env python3
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
    main()# visualize_embeddings.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def visualize_embeddings(original, modified, method="pca"):
    """
    Visualizes original vs. modified embeddings using PCA or t-SNE.
    Args:
        original: numpy array of shape [n, d]
        modified: numpy array of shape [n, d]
        method: "pca" or "tsne"
    """
    if method == "pca":
        reducer = PCA(n_components=2)
    elif method == "tsne":
        reducer = TSNE(n_components=2, perplexity=30, n_iter=300)
    else:
        raise ValueError("Unsupported method: choose 'pca' or 'tsne'")
    
    orig_2d = reducer.fit_transform(original)
    mod_2d = reducer.fit_transform(modified)
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(orig_2d[:, 0], orig_2d[:, 1], c="blue", alpha=0.6)
    plt.title("Original FinBERT Embeddings")
    
    plt.subplot(1, 2, 2)
    plt.scatter(mod_2d[:, 0], mod_2d[:, 1], c="red", alpha=0.6)
    plt.title("Contradiction-Modified Embeddings")
    plt.show()

if __name__ == "__main__":
    # Load saved embeddings from a .npz file.
    data = np.load("./training_data/dataset.npz")
    original = data["finbert_embeddings"]
    
    # For demonstration, we simulate modified embeddings by a simple transformation.
    modified = np.tanh(original)  # In practice, use your contradiction engine output.
    
    visualize_embeddings(original, modified, method="pca")
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
