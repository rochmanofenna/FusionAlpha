"""
FusionAlpha - Contradiction-aware trading signal fusion system.

This package provides:
- FusionNet models for multi-modal feature fusion
- Contradiction detection and routing engines
- Live trading signal generation

Core components:
- models.fusionnet: Multi-modal fusion neural network
- pipelines.contradiction_engine: Adaptive contradiction detection
- routers: Live signal routing and prediction
"""

from .models.fusionnet import FusionNet
from .pipelines.contradiction_engine import AdaptiveContradictionEngine

# Export canonical names for compatibility
ContradictionEngine = AdaptiveContradictionEngine

__version__ = "0.1.0"
__all__ = [
    "FusionNet",
    "AdaptiveContradictionEngine", 
    "ContradictionEngine"
]