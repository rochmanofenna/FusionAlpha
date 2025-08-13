#!/usr/bin/env python3
import time
import logging
import argparse
import numpy as np
import torch
import joblib
import yfinance as yf
from datetime import datetime
from fusion_alpha.models.fusionnet import FusionNet
from fusion_alpha.pipelines.contradiction_engine import AdaptiveContradictionEngine as ContradictionEngine

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
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
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
        run_live_trading()