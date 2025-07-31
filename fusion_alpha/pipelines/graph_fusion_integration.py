"""
Graph-Fusion Integration Pipeline

Integrates PyTorch Geometric contradiction graphs with existing FusionNet architecture.
This bridge connects:
1. Contradiction graph encoding (PyG) -> produces graph embeddings + push-out symbols
2. Existing FusionNet (FinBERT + Technical) -> enhanced with graph context
3. ContradictionEngine -> now graph-aware for better threshold learning

Key enhancements:
- Graph embeddings (z_t) from contradiction analysis 
- Push-out symbols (p) for category-theoretic context
- Enhanced fusion with three-modal input: FinBERT ⊕ Technical ⊕ Graph
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional

# Import our new contradiction graph
from ..models.contradiction_graph import ContradictionGNN, build_contradiction_graph_from_batch

# Import existing components
from ..models.fusionnet import FusionNet
from ..pipelines.contradiction_engine import ContradictionEngine

class GraphAwareFusionNet(nn.Module):
    """
    Enhanced FusionNet that incorporates contradiction graph embeddings
    
    Three-modal fusion: FinBERT ⊕ Technical ⊕ Graph
    Plus push-out symbols for category-theoretic context
    """
    
    def __init__(self, tech_input_dim: int, hidden_dim: int = 512, 
                 output_dim: int = 1, graph_embedding_dim: int = 128,
                 pushout_symbol_dim: int = 32, use_attention: bool = True,
                 target_mode: str = "normalized", dropout_prob: float = 0.5):
        super().__init__()
        
        self.tech_input_dim = tech_input_dim
        self.graph_embedding_dim = graph_embedding_dim
        self.pushout_symbol_dim = pushout_symbol_dim
        self.target_mode = target_mode
        self.dropout_prob = dropout_prob
        
        # Calculate total input dimension for fusion
        # FinBERT (768) + Technical (tech_input_dim) + Graph (128) + Push-out (32)
        total_input_dim = 768 + tech_input_dim + graph_embedding_dim + pushout_symbol_dim
        
        # Core fusion layers
        self.input_projection = nn.Linear(total_input_dim, hidden_dim)
        
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_dim, num_heads=8, batch_first=True, dropout=dropout_prob
            )
        else:
            self.attention = None
            
        # Enhanced fusion network
        self.fusion_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(), 
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # Category-theoretic consistency layer
        self.pushout_consistency = nn.Sequential(
            nn.Linear(pushout_symbol_dim, pushout_symbol_dim),
            nn.Tanh(),  # Ensure bounded push-out symbols
            nn.Linear(pushout_symbol_dim, pushout_symbol_dim)
        )
        
        # Apply Xavier initialization
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, finbert_emb: torch.Tensor, tech_features: torch.Tensor,
                graph_embeddings: torch.Tensor, pushout_symbols: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with four-modal fusion
        
        Args:
            finbert_emb: [batch, 768] FinBERT semantic embeddings
            tech_features: [batch, tech_dim] Technical indicators
            graph_embeddings: [batch, 128] Contradiction graph embeddings  
            pushout_symbols: [batch, 32] Category-theoretic context symbols
            
        Returns:
            predictions: [batch, 1] Trading signal predictions
        """
        batch_size = finbert_emb.size(0)
        
        # Ensure push-out symbols maintain category-theoretic properties
        pushout_consistent = self.pushout_consistency(pushout_symbols)
        
        # Four-modal concatenation: FinBERT ⊕ Technical ⊕ Graph ⊕ Push-out
        fused_input = torch.cat([
            finbert_emb,           # Semantic information
            tech_features,         # Technical signals
            graph_embeddings,      # Contradiction relationships  
            pushout_consistent     # Category-theoretic context
        ], dim=-1)
        
        # Check for NaN values (safety check)
        if torch.isnan(fused_input).any():
            print("⚠️ NaN detected in fused input")
            fused_input = torch.nan_to_num(fused_input, nan=0.0)
        
        # Project to hidden dimension
        h = self.input_projection(fused_input)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout_prob, training=True)  # MC Dropout always on
        
        # Optional attention mechanism for sequence modeling
        if self.attention is not None:
            # Reshape for attention: [batch, seq_len=1, hidden_dim]
            h_seq = h.unsqueeze(1)
            h_attn, _ = self.attention(h_seq, h_seq, h_seq)
            h = h_attn.squeeze(1)
        
        # Final fusion and prediction
        output = self.fusion_layers(h)
        
        # Apply target-mode specific activations
        if self.target_mode == "binary" and not self.training:
            output = torch.clamp(output, min=-10, max=10)  # Prevent overflow
            output = torch.sigmoid(output)
        
        return output.view(-1)

class GraphAwareContradictionEngine(nn.Module):
    """
    Enhanced ContradictionEngine that uses graph embeddings for better threshold learning
    """
    
    def __init__(self, embedding_dim: int = 768, graph_embedding_dim: int = 128):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.graph_embedding_dim = graph_embedding_dim
        
        # Learnable thresholds (enhanced with graph context)
        self.pos_sent_thresh = nn.Parameter(torch.tensor(0.1))
        self.neg_sent_thresh = nn.Parameter(torch.tensor(-0.1))
        self.drop_thresh = nn.Parameter(torch.tensor(0.01))
        self.rise_thresh = nn.Parameter(torch.tensor(0.01))
        
        # Graph-aware threshold adjustment
        self.graph_threshold_modulator = nn.Sequential(
            nn.Linear(graph_embedding_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 4),  # Outputs adjustments for 4 thresholds
            nn.Tanh()  # Bounded adjustments [-1, 1]
        )
        
        # Enhanced embedding transformation with graph context
        self.transform = nn.Sequential(
            nn.Linear(embedding_dim + graph_embedding_dim, embedding_dim),
            nn.Tanh()
        )
    
    def forward(self, finbert_embedding: torch.Tensor, 
                technical_features: torch.Tensor,
                price_movement: torch.Tensor, 
                news_sentiment_score: torch.Tensor,
                graph_embedding: torch.Tensor) -> Tuple[torch.Tensor, Optional[str]]:
        """
        Enhanced contradiction detection with graph context
        
        Args:
            finbert_embedding: [768] Semantic embedding
            technical_features: [tech_dim] Technical indicators
            price_movement: [] Price movement scalar
            news_sentiment_score: [] Sentiment scalar  
            graph_embedding: [128] Contradiction graph embedding
            
        Returns:
            updated_embedding: Enhanced semantic embedding
            contradiction_type: Detected contradiction type
        """
        # Convert tensors to scalars if needed
        sentiment = news_sentiment_score.item() if news_sentiment_score.dim() == 0 else news_sentiment_score
        movement = price_movement.item() if price_movement.dim() == 0 else price_movement
        
        # Graph-aware threshold adjustment
        threshold_adjustments = self.graph_threshold_modulator(graph_embedding)
        
        # Adjust thresholds based on graph context
        pos_thresh_adj = self.pos_sent_thresh + 0.1 * threshold_adjustments[0] 
        neg_thresh_adj = self.neg_sent_thresh + 0.1 * threshold_adjustments[1]
        drop_thresh_adj = self.drop_thresh + 0.01 * threshold_adjustments[2]
        rise_thresh_adj = self.rise_thresh + 0.01 * threshold_adjustments[3]
        
        print("GraphAware ContradictionEngine Debug:")
        print(f"  Sentiment: {sentiment:.4f}, Price Movement: {movement:.4f}")
        print(f"  Adjusted thresholds -> pos: {pos_thresh_adj:.4f}, neg: {neg_thresh_adj:.4f}")
        print(f"  Graph embedding norm: {torch.norm(graph_embedding):.4f}")
        
        # Enhanced contradiction detection with adjusted thresholds
        contradiction_detected = False
        contradiction_type = None
        
        if sentiment > pos_thresh_adj and movement < -drop_thresh_adj:
            contradiction_detected = True
            contradiction_type = "overhype"
        elif sentiment < neg_thresh_adj and movement > rise_thresh_adj:
            contradiction_detected = True
            contradiction_type = "underhype"
        
        if contradiction_detected:
            print(f"  ✅ Contradiction detected: {contradiction_type}")
            # Enhanced transformation with graph context
            enhanced_input = torch.cat([finbert_embedding, graph_embedding])
            updated_embedding = self.transform(enhanced_input)
            return updated_embedding, contradiction_type
        else:
            print("  ⚪ No contradiction detected")
            return finbert_embedding, None

class CompletePipeline(nn.Module):
    """
    Complete integrated pipeline combining:
    1. Contradiction Graph Analysis (PyG)
    2. Graph-Aware Fusion Network  
    3. Enhanced Contradiction Engine
    
    This represents the full BICEP -> ENN -> Fusion Alpha chain
    """
    
    def __init__(self, tech_input_dim: int, hidden_dim: int = 512,
                 target_mode: str = "normalized"):
        super().__init__()
        
        self.tech_input_dim = tech_input_dim
        self.target_mode = target_mode
        
        # Core components
        self.contradiction_gnn = ContradictionGNN(
            embedding_dim=772,  # 768 FinBERT + 4 features
            hidden_dim=256,
            output_dim=128,
            num_layers=3
        )
        
        self.graph_aware_fusion = GraphAwareFusionNet(
            tech_input_dim=tech_input_dim,
            hidden_dim=hidden_dim,
            target_mode=target_mode
        )
        
        self.enhanced_contradiction_engine = GraphAwareContradictionEngine()
        
    def forward(self, finbert_embeddings: torch.Tensor,
                tech_features: torch.Tensor,
                price_movements: torch.Tensor, 
                sentiment_scores: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Complete forward pass through integrated pipeline
        
        Returns dictionary with:
        - predictions: Trading signal predictions
        - graph_embeddings: Contradiction graph representations
        - pushout_symbols: Category-theoretic context symbols
        - contradiction_types: Detected contradiction types per sample
        """
        batch_size = finbert_embeddings.size(0)
        
        # 1. Build contradiction graphs and extract embeddings
        batch_data = []
        for i in range(batch_size):
            sample = {
                'finbert_emb': finbert_embeddings[i],
                'tech_features': tech_features[i],
                'price_movement': price_movements[i].item(),
                'sentiment_score': sentiment_scores[i].item(),
                'concept_label': f'sample_{i}'
            }
            batch_data.append(sample)
        
        # Forward through contradiction GNN
        graph_embeddings, pushout_symbols = self.contradiction_gnn(batch_data)
        
        # 2. Enhanced contradiction detection (optional, for routing)
        contradiction_types = []
        enhanced_embeddings = []
        
        for i in range(batch_size):
            enhanced_emb, ctype = self.enhanced_contradiction_engine(
                finbert_embeddings[i],
                tech_features[i], 
                price_movements[i],
                sentiment_scores[i],
                graph_embeddings[i]
            )
            enhanced_embeddings.append(enhanced_emb)
            contradiction_types.append(ctype if ctype else "none")
        
        enhanced_embeddings = torch.stack(enhanced_embeddings)
        
        # 3. Final fusion prediction
        predictions = self.graph_aware_fusion(
            enhanced_embeddings,
            tech_features,
            graph_embeddings,
            pushout_symbols  
        )
        
        return {
            'predictions': predictions,
            'graph_embeddings': graph_embeddings,
            'pushout_symbols': pushout_symbols,
            'contradiction_types': contradiction_types,
            'enhanced_embeddings': enhanced_embeddings
        }
    
    def get_interpretability_summary(self, results: Dict[str, torch.Tensor]) -> Dict:
        """
        Generate interpretability summary for regulatory compliance
        
        Returns human-readable explanation of model decisions
        """
        batch_size = results['predictions'].size(0)
        summaries = []
        
        for i in range(batch_size):
            summary = {
                'sample_id': i,
                'prediction': results['predictions'][i].item(),
                'contradiction_type': results['contradiction_types'][i],
                'graph_embedding_norm': torch.norm(results['graph_embeddings'][i]).item(),
                'pushout_symbol_norm': torch.norm(results['pushout_symbols'][i]).item(),
                'confidence': abs(results['predictions'][i].item()),
                'explanation': f"Detected {results['contradiction_types'][i]} contradiction with graph strength {torch.norm(results['graph_embeddings'][i]):.3f}"
            }
            summaries.append(summary)
        
        return summaries

# Training utilities for the integrated pipeline

def train_integrated_pipeline(pipeline: CompletePipeline,
                            dataloader: torch.utils.data.DataLoader,
                            optimizer: torch.optim.Optimizer,
                            criterion: nn.Module,
                            num_epochs: int = 10,
                            device: str = 'cuda') -> Dict[str, List[float]]:
    """
    Training loop for the complete integrated pipeline
    """
    pipeline = pipeline.to(device)
    pipeline.train()
    
    training_history = {
        'total_loss': [],
        'prediction_loss': [],
        'graph_consistency_loss': []
    }
    
    for epoch in range(num_epochs):
        epoch_total_loss = 0.0
        epoch_pred_loss = 0.0
        epoch_graph_loss = 0.0
        
        for batch_idx, batch in enumerate(dataloader):
            # Unpack batch
            finbert_embs = batch['finbert_embeddings'].to(device)
            tech_features = batch['tech_features'].to(device)
            price_movements = batch['price_movements'].to(device)
            sentiment_scores = batch['sentiment_scores'].to(device)
            targets = batch['targets'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            results = pipeline(finbert_embs, tech_features, price_movements, sentiment_scores)
            
            # Prediction loss
            pred_loss = criterion(results['predictions'], targets)
            
            # Graph consistency loss (category-theoretic regularization)
            pushout_norm = torch.norm(results['pushout_symbols'], dim=1)
            graph_consistency_loss = 0.01 * torch.mean((pushout_norm - 1.0) ** 2)  # Encourage unit norm
            
            # Total loss
            total_loss = pred_loss + graph_consistency_loss
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(pipeline.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Accumulate losses
            epoch_total_loss += total_loss.item()
            epoch_pred_loss += pred_loss.item()
            epoch_graph_loss += graph_consistency_loss.item()
        
        # Average losses for epoch
        avg_total_loss = epoch_total_loss / len(dataloader)
        avg_pred_loss = epoch_pred_loss / len(dataloader)
        avg_graph_loss = epoch_graph_loss / len(dataloader)
        
        training_history['total_loss'].append(avg_total_loss)
        training_history['prediction_loss'].append(avg_pred_loss)
        training_history['graph_consistency_loss'].append(avg_graph_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Total Loss: {avg_total_loss:.6f}")
        print(f"  Prediction Loss: {avg_pred_loss:.6f}")
        print(f"  Graph Consistency Loss: {avg_graph_loss:.6f}")
    
    return training_history

if __name__ == "__main__":
    # Test the integrated pipeline
    print("Testing Integrated Graph-Fusion Pipeline...")
    
    # Create sample data
    batch_size = 8
    tech_dim = 10
    
    finbert_embs = torch.randn(batch_size, 768)
    tech_features = torch.randn(batch_size, tech_dim)
    price_movements = torch.randn(batch_size) * 0.05  # ±5% movements
    sentiment_scores = torch.randn(batch_size) * 0.8  # ±0.8 sentiment
    
    # Create complete pipeline
    pipeline = CompletePipeline(tech_input_dim=tech_dim, target_mode="normalized")
    
    print(f"📊 Pipeline created with {sum(p.numel() for p in pipeline.parameters()):,} parameters")
    
    # Forward pass
    with torch.no_grad():
        results = pipeline(finbert_embs, tech_features, price_movements, sentiment_scores)
    
    print(f"✅ Predictions shape: {results['predictions'].shape}")
    print(f"✅ Graph embeddings shape: {results['graph_embeddings'].shape}")
    print(f"✅ Push-out symbols shape: {results['pushout_symbols'].shape}")
    print(f"✅ Detected contradictions: {results['contradiction_types']}")
    
    # Test interpretability
    interpretability = pipeline.get_interpretability_summary(results)
    print(f"✅ Sample interpretation: {interpretability[0]}")
    
    print("🚀 Integrated Pipeline Implementation Complete!")
    print("📈 Ready for integration with BICEP and ENN components!")