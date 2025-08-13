#!/usr/bin/env python3
"""
Unified Pipeline Integration Layer

Complete integration of BICEP + ENN + FusionAlpha according to the architecture:
1. Contradiction Graph Encoder → PyG MessagePassing → z_t
2. BICEP + ENN → Brownian paths → state collapse → p_t  
3. Feature Stack: x_t = [z_t || p_t || FinBERT || Technical Analysis]
4. Fusion Alpha → (direction, raw_size)
5. Risk Dial (Ising/limit-colimit) → leverage_mult
6. Position Sizing: size = raw_size × leverage_mult
7. Live Router/Execution
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from dataclasses import dataclass
import json

# Add parent paths for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import pipeline components
from backends.bicep_integration import UnderhypeBICEPSimulator
from enn.bicep_adapter import BICEPDimensionAdapter, CleanBICEPLayer
from enn.model import ENNModelWithSparsityControl
from enn.config import Config as ENNConfig
from fusion_alpha.models.contradiction_graph import ContradictionGNN
from fusion_alpha.models.fusionnet import FusionNet
from fusion_alpha.pipelines.contradiction_engine import ContradictionEngine
from fusion_alpha.models.real_finbert import RealFinBERTProcessor
from core.underhype_engine import UnderhypeEngine

logger = logging.getLogger(__name__)

@dataclass
class UnifiedSignal:
    """Complete signal with all pipeline outputs"""
    ticker: str
    timestamp: datetime
    # Graph encoder output
    z_t: torch.Tensor  # Graph embedding
    # BICEP+ENN output  
    p_t: torch.Tensor  # Collapsed state representation
    # FinBERT output
    sentiment_score: float
    sentiment_embedding: torch.Tensor
    # Technical indicators
    technical_features: Dict[str, float]
    # FusionAlpha output
    direction: str  # 'buy', 'sell', 'hold'
    raw_size: float  # Base position size
    confidence: float
    # Risk management
    leverage_mult: float  # Risk dial output
    final_size: float  # raw_size × leverage_mult
    # Signal metadata
    signal_type: str  # 'underhype', 'overhype', 'normal'
    headline: str
    expected_return: float

class UnifiedPipelineIntegration:
    """
    Complete three-pipeline integration system
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._get_default_config()
        self.device = torch.device(self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Initialize all pipeline components
        self._init_components()
        
        logger.info(f"✅ Unified Pipeline Integration initialized on {self.device}")
        logger.info(f"   - BICEP: {'Enabled' if self.bicep_simulator else 'Disabled'}")
        logger.info(f"   - ENN: {'Enabled' if self.enn_model else 'Disabled'}")
        logger.info(f"   - Graph Encoder: {'Enabled' if self.graph_encoder else 'Disabled'}")
        logger.info(f"   - FusionAlpha: Enabled")
    
    def _get_default_config(self) -> Dict:
        """Default configuration for all pipelines"""
        return {
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'enable_bicep': True,
            'enable_enn': True,
            'enable_graph': True,
            # BICEP config
            'bicep': {
                'n_paths': 100,
                'n_steps': 50,
                'scenarios_per_ticker': 20
            },
            # ENN config
            'enn': {
                'num_neurons': 128,
                'num_states': 8,
                'entanglement_dim': 16,
                'memory_length': 10,
                'dropout_rate': 0.1
            },
            # Graph config
            'graph': {
                'node_dim': 64,
                'edge_dim': 32,
                'hidden_dim': 128,
                'num_layers': 3
            },
            # Risk management
            'risk': {
                'max_leverage': 3.0,
                'base_position_size': 0.02,
                'volatility_adjustment': True
            }
        }
    
    def _init_components(self):
        """Initialize all pipeline components"""
        
        # 1. BICEP Simulator
        if self.config.get('enable_bicep', True):
            try:
                self.bicep_simulator = UnderhypeBICEPSimulator(device=str(self.device))
                logger.info("✅ BICEP simulator initialized")
            except Exception as e:
                logger.warning(f"⚠️ BICEP initialization failed: {e}")
                self.bicep_simulator = None
        else:
            self.bicep_simulator = None
        
        # 2. ENN Model with BICEP integration
        if self.config.get('enable_enn', True):
            try:
                enn_config = ENNConfig(**self.config['enn'])
                self.enn_model = ENNModelWithSparsityControl(enn_config)
                
                # Add BICEP adapter
                if self.bicep_simulator:
                    self.bicep_adapter = BICEPDimensionAdapter(
                        input_dim=enn_config.num_states,
                        output_dim=enn_config.num_neurons * enn_config.num_states,
                        n_paths=self.config['bicep']['n_paths'],
                        n_steps=self.config['bicep']['n_steps'],
                        device=str(self.device)
                    )
                    self.bicep_adapter.to(self.device)
                
                self.enn_model.to(self.device)
                logger.info("✅ ENN model initialized")
            except Exception as e:
                logger.warning(f"⚠️ ENN initialization failed: {e}")
                self.enn_model = None
                self.bicep_adapter = None
        else:
            self.enn_model = None
            self.bicep_adapter = None
        
        # 3. Graph Encoder
        if self.config.get('enable_graph', True):
            try:
                self.graph_encoder = ContradictionGNN(
                    node_dim=self.config['graph']['node_dim'],
                    edge_dim=self.config['graph']['edge_dim'],
                    hidden_dim=self.config['graph']['hidden_dim'],
                    num_layers=self.config['graph']['num_layers']
                )
                self.graph_encoder.to(self.device)
                logger.info("✅ Graph encoder initialized")
            except Exception as e:
                logger.warning(f"⚠️ Graph encoder initialization failed: {e}")
                self.graph_encoder = None
        else:
            self.graph_encoder = None
        
        # 4. FinBERT Processor
        try:
            self.finbert_processor = RealFinBERTProcessor()
            logger.info("✅ FinBERT processor initialized")
        except Exception as e:
            logger.warning(f"⚠️ FinBERT initialization failed: {e}")
            self.finbert_processor = None
        
        # 5. FusionAlpha Components
        self.contradiction_engine = ContradictionEngine()
        self.fusion_net = FusionNet(
            input_dim=768 + 10,  # FinBERT + technical features
            hidden_dim=256
        ).to(self.device)
        
        # 6. Underhype Engine
        self.underhype_engine = UnderhypeEngine(confidence_threshold=2.5)
        
        # 7. Risk Dial
        self.risk_dial = RiskDial(self.config['risk'])
    
    def process_market_data(self, market_data: Dict) -> UnifiedSignal:
        """
        Process market data through complete pipeline
        
        Args:
            market_data: Dict with keys:
                - ticker: str
                - headline: str
                - price_data: pd.DataFrame
                - graph_data: Optional[Dict]  # For graph encoder
                
        Returns:
            UnifiedSignal with all pipeline outputs
        """
        
        ticker = market_data['ticker']
        headline = market_data['headline']
        price_data = market_data['price_data']
        
        # 1. Graph Encoder (z_t)
        if self.graph_encoder and market_data.get('graph_data'):
            z_t = self._process_graph_data(market_data['graph_data'])
        else:
            # Fallback: zero vector
            z_t = torch.zeros(self.config['graph']['hidden_dim'], device=self.device)
        
        # 2. BICEP + ENN (p_t)
        p_t = self._process_bicep_enn(ticker, price_data)
        
        # 3. FinBERT sentiment
        sentiment_score, sentiment_embedding = self._process_sentiment(headline)
        
        # 4. Technical indicators
        technical_features = self._calculate_technical_indicators(price_data)
        
        # 5. Feature stack: x_t = [z_t || p_t || FinBERT || Technical]
        x_t = self._create_feature_stack(z_t, p_t, sentiment_embedding, technical_features)
        
        # 6. FusionAlpha
        direction, raw_size, confidence = self._process_fusion_alpha(x_t, sentiment_score)
        
        # 7. Risk Dial
        leverage_mult = self.risk_dial.calculate_leverage(
            ticker=ticker,
            confidence=confidence,
            volatility=technical_features.get('volatility', 0.02),
            market_conditions=self._assess_market_conditions(price_data)
        )
        
        # 8. Final position sizing
        final_size = raw_size * leverage_mult
        
        # Determine signal type
        price_movement = technical_features.get('daily_return', 0)
        signal_type = self._determine_signal_type(sentiment_score, price_movement)
        
        # Expected return based on historical analysis
        expected_return = self.underhype_engine.thresholds.get(
            ticker, self.underhype_engine.thresholds['DEFAULT']
        )['expected_return']
        
        return UnifiedSignal(
            ticker=ticker,
            timestamp=datetime.now(),
            z_t=z_t,
            p_t=p_t,
            sentiment_score=sentiment_score,
            sentiment_embedding=sentiment_embedding,
            technical_features=technical_features,
            direction=direction,
            raw_size=raw_size,
            confidence=confidence,
            leverage_mult=leverage_mult,
            final_size=final_size,
            signal_type=signal_type,
            headline=headline,
            expected_return=expected_return
        )
    
    def _process_graph_data(self, graph_data: Dict) -> torch.Tensor:
        """Process graph data through encoder"""
        try:
            # Convert graph data to PyG format
            # This is a placeholder - actual implementation depends on graph structure
            node_features = torch.tensor(graph_data['node_features'], device=self.device)
            edge_index = torch.tensor(graph_data['edge_index'], device=self.device)
            edge_attr = torch.tensor(graph_data.get('edge_attr', []), device=self.device)
            
            # Process through graph encoder
            z_t = self.graph_encoder(node_features, edge_index, edge_attr)
            
            return z_t
        except Exception as e:
            logger.warning(f"Graph processing failed: {e}")
            return torch.zeros(self.config['graph']['hidden_dim'], device=self.device)
    
    def _process_bicep_enn(self, ticker: str, price_data: pd.DataFrame) -> torch.Tensor:
        """Process through BICEP + ENN pipeline"""
        
        if not self.enn_model:
            return torch.zeros(self.config['enn']['num_neurons'], device=self.device)
        
        try:
            # Extract price features
            prices = price_data['close'].values[-20:]  # Last 20 days
            returns = np.diff(np.log(prices))
            
            # Create input tensor
            input_features = torch.tensor(returns[-self.config['enn']['num_states']:], 
                                        dtype=torch.float32, device=self.device)
            
            # Add batch dimension
            input_features = input_features.unsqueeze(0)
            
            # Process through ENN
            enn_output = self.enn_model(input_features)
            
            # If BICEP adapter available, enhance with BICEP
            if self.bicep_adapter:
                bicep_enhanced = self.bicep_adapter(input_features)
                # Combine ENN and BICEP outputs
                p_t = (enn_output.flatten() + bicep_enhanced.flatten()) / 2
            else:
                p_t = enn_output.flatten()
            
            return p_t
            
        except Exception as e:
            logger.warning(f"BICEP+ENN processing failed: {e}")
            return torch.zeros(self.config['enn']['num_neurons'], device=self.device)
    
    def _process_sentiment(self, headline: str) -> Tuple[float, torch.Tensor]:
        """Process headline through FinBERT"""
        
        if not self.finbert_processor:
            # Fallback values
            return 0.0, torch.zeros(768, device=self.device)
        
        try:
            result = self.finbert_processor.process_text(headline)
            sentiment_score = result['sentiment_score']
            
            # Get embedding (last hidden state mean)
            embedding = torch.tensor(result['embedding'], device=self.device)
            
            return sentiment_score, embedding
            
        except Exception as e:
            logger.warning(f"Sentiment processing failed: {e}")
            return 0.0, torch.zeros(768, device=self.device)
    
    def _calculate_technical_indicators(self, price_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate technical indicators"""
        
        try:
            close_prices = price_data['close'].values
            
            # Basic indicators
            daily_return = (close_prices[-1] / close_prices[-2]) - 1 if len(close_prices) > 1 else 0
            volatility = np.std(np.diff(np.log(close_prices))) if len(close_prices) > 2 else 0.02
            
            # Moving averages
            sma_20 = np.mean(close_prices[-20:]) if len(close_prices) >= 20 else close_prices[-1]
            sma_50 = np.mean(close_prices[-50:]) if len(close_prices) >= 50 else sma_20
            
            # RSI
            rsi = self._calculate_rsi(close_prices)
            
            # MACD
            macd, signal = self._calculate_macd(close_prices)
            
            return {
                'daily_return': daily_return,
                'volatility': volatility,
                'sma_20': sma_20,
                'sma_50': sma_50,
                'price_to_sma20': close_prices[-1] / sma_20,
                'price_to_sma50': close_prices[-1] / sma_50,
                'rsi': rsi,
                'macd': macd,
                'macd_signal': signal,
                'volume_ratio': price_data['volume'].iloc[-1] / price_data['volume'].mean() if 'volume' in price_data else 1.0
            }
            
        except Exception as e:
            logger.warning(f"Technical indicator calculation failed: {e}")
            return {
                'daily_return': 0,
                'volatility': 0.02,
                'sma_20': 100,
                'sma_50': 100,
                'price_to_sma20': 1.0,
                'price_to_sma50': 1.0,
                'rsi': 50,
                'macd': 0,
                'macd_signal': 0,
                'volume_ratio': 1.0
            }
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI"""
        if len(prices) < period + 1:
            return 50.0
            
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
            
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_macd(self, prices: np.ndarray) -> Tuple[float, float]:
        """Calculate MACD"""
        if len(prices) < 26:
            return 0.0, 0.0
            
        # Simple EMA calculation
        ema_12 = self._ema(prices, 12)
        ema_26 = self._ema(prices, 26)
        
        macd = ema_12 - ema_26
        signal = self._ema(np.array([macd]), 9)
        
        return macd, signal
    
    def _ema(self, data: np.ndarray, period: int) -> float:
        """Calculate EMA"""
        if len(data) < period:
            return data[-1] if len(data) > 0 else 0
            
        multiplier = 2 / (period + 1)
        ema = data[-period]
        
        for price in data[-period+1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
            
        return ema
    
    def _create_feature_stack(self, z_t: torch.Tensor, p_t: torch.Tensor, 
                            sentiment_embedding: torch.Tensor, 
                            technical_features: Dict[str, float]) -> torch.Tensor:
        """Create unified feature vector"""
        
        # Convert technical features to tensor
        tech_values = torch.tensor([
            technical_features['daily_return'],
            technical_features['volatility'],
            technical_features['price_to_sma20'],
            technical_features['price_to_sma50'],
            technical_features['rsi'] / 100.0,
            technical_features['macd'],
            technical_features['macd_signal'],
            technical_features['volume_ratio'],
            0.0,  # Placeholder
            0.0   # Placeholder
        ], device=self.device)
        
        # Concatenate all features
        # Ensure all tensors are 1D
        z_t_flat = z_t.flatten()
        p_t_flat = p_t.flatten()
        sentiment_flat = sentiment_embedding.flatten()
        
        # Create fixed-size representations
        z_t_fixed = z_t_flat[:128] if len(z_t_flat) >= 128 else torch.cat([
            z_t_flat, torch.zeros(128 - len(z_t_flat), device=self.device)
        ])
        
        p_t_fixed = p_t_flat[:128] if len(p_t_flat) >= 128 else torch.cat([
            p_t_flat, torch.zeros(128 - len(p_t_flat), device=self.device)
        ])
        
        sentiment_fixed = sentiment_flat[:768] if len(sentiment_flat) >= 768 else torch.cat([
            sentiment_flat, torch.zeros(768 - len(sentiment_flat), device=self.device)
        ])
        
        # Stack features: [z_t || p_t || FinBERT || Technical]
        x_t = torch.cat([z_t_fixed, p_t_fixed, sentiment_fixed, tech_values])
        
        return x_t
    
    def _process_fusion_alpha(self, x_t: torch.Tensor, 
                            sentiment_score: float) -> Tuple[str, float, float]:
        """Process through FusionAlpha"""
        
        try:
            # Add batch dimension
            x_t_batch = x_t.unsqueeze(0)
            
            # Forward through FusionNet
            with torch.no_grad():
                output = self.fusion_net(x_t_batch)
            
            # Extract predictions
            direction_logits = output[:, :3]  # First 3 for direction
            size_output = output[:, 3]  # 4th for size
            confidence_output = output[:, 4] if output.shape[1] > 4 else torch.tensor([0.5])
            
            # Get direction
            direction_idx = torch.argmax(direction_logits).item()
            directions = ['sell', 'hold', 'buy']
            direction = directions[direction_idx]
            
            # Get size and confidence
            raw_size = torch.sigmoid(size_output).item() * self.config['risk']['base_position_size']
            confidence = torch.sigmoid(confidence_output).item()
            
            return direction, raw_size, confidence
            
        except Exception as e:
            logger.warning(f"FusionAlpha processing failed: {e}")
            # Fallback based on sentiment
            if sentiment_score < -0.1:
                return 'buy', self.config['risk']['base_position_size'], 0.5
            else:
                return 'hold', 0.0, 0.0
    
    def _determine_signal_type(self, sentiment: float, price_movement: float) -> str:
        """Determine signal type based on contradiction theory"""
        
        if sentiment < -0.1 and price_movement > 0.02:
            return 'underhype'
        elif sentiment > 0.1 and price_movement < -0.02:
            return 'overhype'
        else:
            return 'normal'
    
    def _assess_market_conditions(self, price_data: pd.DataFrame) -> Dict[str, float]:
        """Assess overall market conditions"""
        
        try:
            prices = price_data['close'].values
            returns = np.diff(np.log(prices))
            
            return {
                'volatility': np.std(returns) if len(returns) > 0 else 0.02,
                'trend': np.mean(returns[-5:]) if len(returns) >= 5 else 0,
                'regime': 'normal'  # Could be enhanced with regime detection
            }
        except:
            return {
                'volatility': 0.02,
                'trend': 0,
                'regime': 'normal'
            }

class RiskDial:
    """
    Risk management system using limit/colimit approach
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.max_leverage = config['max_leverage']
        self.base_position_size = config['base_position_size']
        
    def calculate_leverage(self, ticker: str, confidence: float, 
                         volatility: float, market_conditions: Dict) -> float:
        """
        Calculate leverage multiplier based on multiple factors
        
        Uses a limit/colimit inspired approach where multiple risk factors
        are combined to determine final leverage
        """
        
        # Base leverage from confidence
        confidence_leverage = 1.0 + (confidence - 0.5) * 2.0  # 0-2x based on confidence
        
        # Volatility adjustment (inverse relationship)
        vol_adjustment = np.exp(-volatility * 10)  # Lower leverage for high volatility
        
        # Market regime adjustment
        regime = market_conditions.get('regime', 'normal')
        regime_mult = {
            'bullish': 1.2,
            'normal': 1.0,
            'bearish': 0.8,
            'crisis': 0.5
        }.get(regime, 1.0)
        
        # Combine factors (limit operation)
        raw_leverage = confidence_leverage * vol_adjustment * regime_mult
        
        # Apply bounds
        final_leverage = np.clip(raw_leverage, 0.5, self.max_leverage)
        
        return final_leverage

def create_monitoring_dashboard():
    """Create enhanced monitoring dashboard for unified pipeline"""
    
    dashboard_config = {
        'title': 'Unified Pipeline Monitoring',
        'sections': [
            {
                'name': 'BICEP Performance',
                'metrics': ['throughput', 'latency', 'gpu_utilization']
            },
            {
                'name': 'ENN State',
                'metrics': ['active_neurons', 'entanglement_strength', 'memory_usage']
            },
            {
                'name': 'FusionAlpha',
                'metrics': ['signals_generated', 'contradiction_rate', 'accuracy']
            },
            {
                'name': 'Risk Management',
                'metrics': ['current_leverage', 'portfolio_exposure', 'var_95']
            }
        ],
        'update_interval': 1.0,  # seconds
        'retention_period': 3600  # seconds
    }
    
    return dashboard_config

# Example usage
if __name__ == "__main__":
    # Initialize unified pipeline
    pipeline = UnifiedPipelineIntegration()
    
    # Example market data
    market_data = {
        'ticker': 'AAPL',
        'headline': 'Apple faces regulatory scrutiny over App Store policies',
        'price_data': pd.DataFrame({
            'close': np.random.randn(100).cumsum() + 150,
            'volume': np.random.randint(1000000, 5000000, 100)
        }),
        'graph_data': None  # Would contain actual graph structure
    }
    
    # Process through pipeline
    signal = pipeline.process_market_data(market_data)
    
    print(f"Unified Pipeline Output:")
    print(f"   Ticker: {signal.ticker}")
    print(f"   Signal Type: {signal.signal_type}")
    print(f"   Direction: {signal.direction}")
    print(f"   Confidence: {signal.confidence:.2f}")
    print(f"   Final Size: {signal.final_size:.4f}")
    print(f"   Expected Return: {signal.expected_return:.2f}%")