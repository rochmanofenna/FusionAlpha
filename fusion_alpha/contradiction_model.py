import torch
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
        print("Sample Gate Weights:", gate_weight)