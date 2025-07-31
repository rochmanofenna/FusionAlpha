import numpy as np
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
# You could integrate this TransformerTimeSeriesEncoder into your overall model for time-series modeling.