import torch
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
    print("Fused output shape:", fused_output.shape)