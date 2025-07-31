import networkx as nx
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
    main()