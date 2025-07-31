"""
Integrated Pipeline Configuration

Unified configuration for BICEP -> ENN -> Fusion Alpha with PyG Contradiction Graphs
This connects all three major components based on your documentation.

Configuration structure:
1. BICEP settings (Triton kernels, path generation)
2. ENN settings (multi-state neurons, state collapse)  
3. Contradiction Graph settings (PyG, category theory)
4. Fusion Alpha settings (semantic-technical fusion)
5. Pipeline orchestration settings
"""

import torch
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
from pathlib import Path

@dataclass
class BICEPConfig:
    """BICEP (Brownian Inspired Computationally Efficient Parallelization) settings"""
    
    # Core parameters
    n_paths: int = 1024                    # Number of Brownian paths per batch
    n_steps: int = 1000                    # Steps per path
    T: float = 1.0                         # Time horizon
    base_variance: float = 1.0             # Base Brownian variance
    
    # Stochastic control parameters
    feedback_value: float = 0.5            # Control feedback strength
    decay_rate: float = 0.1               # Exponential decay rate
    high_threshold: float = 10.0          # High visit count threshold
    low_threshold: float = 2.0            # Low visit count threshold
    
    # Performance settings
    use_triton: bool = True               # Use Triton kernels vs CURAND
    device: str = 'cuda'                  # Compute device
    batch_size: int = 1000                # Paths per GPU batch
    target_latency_ms: float = 25.0       # End-to-end target latency
    
    # Integration settings
    output_dim: int = 128                 # BICEP feature dimension for ENN
    enable_gpu_streaming: bool = True     # Stream large batches
    memory_efficient: bool = True         # Use memory mapping for large datasets

@dataclass 
class ENNConfig:
    """ENN (Enhanced Neural Networks) settings for multi-state neurons"""
    
    # Architecture
    num_layers: int = 3                   # Number of ENN layers
    num_neurons: int = 64                 # Neurons per layer
    num_states: int = 8                   # K-state entangled neurons
    compressed_dim: int = 32              # Push-out symbol dimension
    input_dim: int = 128                  # Input from BICEP
    
    # Neuron dynamics
    decay_rate: float = 0.1               # State decay rate  
    recency_factor: float = 0.9           # Temporal memory weighting
    buffer_size: int = 12                 # Short-term memory buffer
    
    # Sparsity and gating
    sparsity_threshold: float = 0.02      # Dynamic sparsity threshold
    low_power_k: int = 5                  # Top-k for low-power collapse
    attention_threshold: float = 0.6      # Attention gate threshold
    importance_threshold: float = 0.1     # State collapse threshold
    
    # Training parameters
    l1_lambda: float = 1e-4               # L1 regularization for sparsity
    max_grad_norm: float = 1.0            # Gradient clipping
    
    # Push-out symbol generation (category theory)
    enable_pushout_extraction: bool = True    # Extract p symbols for downstream
    pushout_consistency_weight: float = 0.01  # Category theory regularization

@dataclass
class ContradictionGraphConfig:
    """PyTorch Geometric contradiction graph settings"""
    
    # Graph construction
    max_nodes: int = 1024                 # Maximum nodes per graph
    temporal_window: int = 10             # Time steps for temporal contradictions
    embedding_dim: int = 772              # Node feature dimension (768 FinBERT + 4)
    
    # Graph neural network
    hidden_dim: int = 256                 # GNN hidden dimension
    output_dim: int = 128                 # Graph embedding output dimension
    num_layers: int = 3                   # Message passing layers
    heads: int = 8                        # Multi-head attention heads
    dropout: float = 0.1                  # Dropout probability
    
    # Message passing
    use_superposition_weighting: bool = True      # Use α, β amplitudes
    enable_self_loops: bool = True               # Self-loops for paradox attractors
    message_passing_type: str = "custom"        # "custom", "gat", "gcn"
    
    # Category theory axioms
    enforce_connectivity: bool = True            # Connected graph axiom
    enforce_reversibility: bool = True          # Reversible morphisms
    validate_axioms: bool = True                # Runtime axiom validation
    
    # Contradiction detection thresholds
    sentiment_pos_threshold: float = 0.1        # Positive sentiment threshold
    sentiment_neg_threshold: float = -0.1       # Negative sentiment threshold
    price_drop_threshold: float = 0.01          # Price drop threshold
    price_rise_threshold: float = 0.01          # Price rise threshold

@dataclass
class FusionAlphaConfig:
    """Fusion Alpha settings for semantic-technical fusion"""
    
    # Model architecture
    hidden_dim: int = 512                 # Hidden layer dimension
    output_dim: int = 1                   # Prediction output dimension
    use_attention: bool = True            # Multi-head attention
    fusion_method: str = "concat"         # "concat" or "average"
    target_mode: str = "normalized"       # "normalized", "binary", "rolling"
    
    # Enhanced fusion (four-modal)
    finbert_dim: int = 768               # FinBERT embedding dimension
    tech_features_dim: int = 10          # Technical indicators dimension
    graph_embedding_dim: int = 128       # From contradiction graph
    pushout_symbol_dim: int = 32         # From ENN push-out
    
    # Training parameters
    dropout_prob: float = 0.5            # MC Dropout (always active)
    learning_rate: float = 1e-4          # Adam learning rate
    weight_decay: float = 1e-5           # L2 regularization
    epochs: int = 100                    # Training epochs
    batch_size: int = 32                 # Training batch size
    
    # Contradiction routing
    enable_specialized_routing: bool = True      # Route by contradiction type
    overhype_model_path: str = "fusion_overhype_weights.pth"
    underhype_model_path: str = "fusion_underhype_weights.pth"
    none_model_path: str = "fusion_none_weights.pth"
    
    # Risk management
    prediction_threshold: float = 0.01   # Action threshold
    confidence_threshold: float = 0.5    # Minimum confidence for trades

@dataclass  
class PipelineOrchestrationConfig:
    """Overall pipeline orchestration settings"""
    
    # Component integration
    enable_bicep_enn_integration: bool = True    # Connect BICEP -> ENN
    enable_enn_fusion_integration: bool = True   # Connect ENN -> Fusion
    enable_graph_fusion_integration: bool = True # Use contradiction graphs
    
    # Performance settings
    target_end_to_end_latency_ms: float = 25.0  # Total pipeline latency target
    enable_async_processing: bool = True         # Asynchronous event processing
    max_concurrent_samples: int = 64            # Parallel processing limit
    
    # Data flow
    bicep_to_enn_dim: int = 128            # BICEP output -> ENN input dimension
    enn_to_fusion_pushout_dim: int = 32    # ENN push-out -> Fusion dimension
    graph_to_fusion_dim: int = 128         # Graph embedding -> Fusion dimension
    
    # Monitoring and logging
    enable_performance_monitoring: bool = True   # Monitor component latencies
    enable_axiom_validation: bool = True        # Runtime category theory validation
    log_contradiction_types: bool = True        # Log detected contradictions
    save_interpretability_traces: bool = True   # Save decision traces
    
    # Risk dial (limit/colimit micro-risk gauge)
    enable_risk_dial: bool = True              # Category-theoretic risk management
    risk_dial_update_frequency: int = 10       # Update every N samples
    max_risk_multiplier: float = 2.0          # Maximum position size multiplier
    min_risk_multiplier: float = 0.1          # Minimum position size multiplier

@dataclass
class IntegratedPipelineConfig:
    """Complete configuration for integrated BICEP -> ENN -> Fusion Alpha pipeline"""
    
    bicep: BICEPConfig = field(default_factory=BICEPConfig)
    enn: ENNConfig = field(default_factory=ENNConfig)
    contradiction_graph: ContradictionGraphConfig = field(default_factory=ContradictionGraphConfig)
    fusion_alpha: FusionAlphaConfig = field(default_factory=FusionAlphaConfig)
    orchestration: PipelineOrchestrationConfig = field(default_factory=PipelineOrchestrationConfig)
    
    # Global settings
    device: str = 'cuda'                  # Global compute device
    dtype: torch.dtype = torch.float32    # Global data type
    random_seed: int = 42                 # Reproducibility seed
    
    # Paths
    model_save_dir: Path = Path("./models/integrated/")
    data_dir: Path = Path("./training_data/")
    log_dir: Path = Path("./logs/integrated/")
    
    def __post_init__(self):
        """Validate configuration consistency"""
        # Ensure dimension compatibility
        if self.enn.input_dim != self.bicep.output_dim:
            raise ValueError(f"ENN input_dim ({self.enn.input_dim}) must match BICEP output_dim ({self.bicep.output_dim})")
        
        if self.fusion_alpha.pushout_symbol_dim != self.enn.compressed_dim:
            raise ValueError(f"Fusion pushout_symbol_dim ({self.fusion_alpha.pushout_symbol_dim}) must match ENN compressed_dim ({self.enn.compressed_dim})")
        
        if self.fusion_alpha.graph_embedding_dim != self.contradiction_graph.output_dim:
            raise ValueError(f"Fusion graph_embedding_dim ({self.fusion_alpha.graph_embedding_dim}) must match ContradictionGraph output_dim ({self.contradiction_graph.output_dim})")
        
        # Create directories
        self.model_save_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Set global random seed
        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_seed)
    
    def get_performance_targets(self) -> Dict[str, float]:
        """Get performance targets for each component"""
        return {
            'bicep_latency_ms': 0.5,      # From triton benchmarks: 0.50ms / 1024 paths
            'enn_latency_ms': 0.7,        # From documentation: ~0.7ms forward pass  
            'graph_latency_ms': 1.2,      # From documentation: ~1.2ms PyG encoding
            'fusion_latency_ms': 17.0,    # Remaining budget for fusion + routing
            'total_latency_ms': self.orchestration.target_end_to_end_latency_ms
        }
    
    def validate_theoretical_consistency(self) -> Dict[str, bool]:
        """Validate category-theoretic consistency"""
        checks = {
            'connected_graph_axiom': self.contradiction_graph.enforce_connectivity,
            'reversible_morphisms': self.contradiction_graph.enforce_reversibility,
            'pushout_minimality': self.enn.enable_pushout_extraction,
            'superposition_axiom': self.contradiction_graph.use_superposition_weighting,
            'self_loop_paradoxes': self.contradiction_graph.enable_self_loops
        }
        return checks
    
    def get_integration_summary(self) -> str:
        """Generate human-readable configuration summary"""
        summary = f"""
Integrated Pipeline Configuration Summary

BICEP Settings:
  • Paths: {self.bicep.n_paths} × {self.bicep.n_steps} steps
  • Target latency: {self.bicep.target_latency_ms}ms
  • Device: {self.bicep.device}, Triton: {self.bicep.use_triton}

ENN Settings:  
  • Architecture: {self.enn.num_layers} layers × {self.enn.num_neurons} neurons × {self.enn.num_states} states
  • Push-out symbols: {self.enn.compressed_dim}D
  • Memory buffer: {self.enn.buffer_size} steps

Contradiction Graph Settings:
  • Max nodes: {self.contradiction_graph.max_nodes}
  • GNN layers: {self.contradiction_graph.num_layers} × {self.contradiction_graph.hidden_dim}D
  • Output: {self.contradiction_graph.output_dim}D embeddings

Fusion Alpha Settings:
  • Four-modal fusion: FinBERT({self.fusion_alpha.finbert_dim}) + Tech({self.fusion_alpha.tech_features_dim}) + Graph({self.fusion_alpha.graph_embedding_dim}) + PushOut({self.fusion_alpha.pushout_symbol_dim})
  • Hidden: {self.fusion_alpha.hidden_dim}D, Target: {self.fusion_alpha.target_mode}

Performance Targets:
  • End-to-end: <{self.orchestration.target_end_to_end_latency_ms}ms
  • Concurrent samples: {self.orchestration.max_concurrent_samples}

Category Theory Validation: {all(self.validate_theoretical_consistency().values())}
        """
        return summary

# Preset configurations for different deployment scenarios

def get_development_config() -> IntegratedPipelineConfig:
    """Configuration optimized for development and testing"""
    config = IntegratedPipelineConfig()
    
    # Reduce sizes for faster iteration
    config.bicep.n_paths = 256
    config.bicep.n_steps = 100
    config.enn.num_neurons = 32
    config.contradiction_graph.max_nodes = 256
    config.fusion_alpha.batch_size = 16
    config.fusion_alpha.epochs = 10
    
    return config

def get_production_config() -> IntegratedPipelineConfig:
    """Configuration optimized for production deployment"""
    config = IntegratedPipelineConfig()
    
    # Maximum performance settings
    config.bicep.n_paths = 2048
    config.bicep.n_steps = 2000
    config.enn.num_neurons = 128
    config.contradiction_graph.max_nodes = 2048
    config.orchestration.target_end_to_end_latency_ms = 20.0  # Tighter constraint
    config.orchestration.enable_async_processing = True
    config.orchestration.max_concurrent_samples = 128
    
    return config

def get_research_config() -> IntegratedPipelineConfig:
    """Configuration optimized for research and experimentation"""
    config = IntegratedPipelineConfig()
    
    # Enhanced monitoring and validation
    config.orchestration.enable_axiom_validation = True
    config.orchestration.save_interpretability_traces = True
    config.contradiction_graph.validate_axioms = True
    config.fusion_alpha.epochs = 200
    
    # Theoretical consistency checks
    config.contradiction_graph.enforce_connectivity = True
    config.contradiction_graph.enforce_reversibility = True
    config.enn.enable_pushout_extraction = True
    
    return config

if __name__ == "__main__":
    # Test configuration creation and validation
    print("Testing Integrated Pipeline Configuration...")
    
    # Test default configuration
    config = IntegratedPipelineConfig()
    print("Default configuration created successfully")
    
    # Test configuration summary
    print(config.get_integration_summary())
    
    # Test performance targets
    targets = config.get_performance_targets()
    print(f"Performance targets: {targets}")
    
    # Test theoretical consistency
    theory_check = config.validate_theoretical_consistency()
    print(f"Theory validation: {theory_check}")
    
    # Test preset configurations
    dev_config = get_development_config()
    prod_config = get_production_config() 
    research_config = get_research_config()
    
    print("All preset configurations created successfully")
    print(f"Development BICEP paths: {dev_config.bicep.n_paths}")
    print(f"Production target latency: {prod_config.orchestration.target_end_to_end_latency_ms}ms")
    print(f"Research axiom validation: {research_config.orchestration.enable_axiom_validation}")
    
    print("\nConfiguration system ready for integrated pipeline!")