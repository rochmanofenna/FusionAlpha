import numpy as np
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
            print("NaN or Inf in tech features. Skipping batch.")
            continue

        if torch.isnan(batch_finbert).any() or torch.isinf(batch_finbert).any():
            print("NaN or Inf in FinBERT embeddings. Skipping batch.")
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
            print("NaN/Inf loss detected. Skipping batch.")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_loss += loss.item() * batch_finbert.size(0)

    avg_loss = epoch_loss / num_samples
    loss_history.append(avg_loss)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")

# Save model
torch.save(model.state_dict(), "./training_data/fusion_net_contradiction_weights.pth")
print("Training complete. Model saved to ./training_data/fusion_net_contradiction_weights.pth")

# Optional: plot loss curve
try:
    import matplotlib.pyplot as plt
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.savefig("./training_data/loss_curve.png")
    print("Loss curve saved to training_data/loss_curve.png")
except ImportError:
    print("matplotlib not installed. Skipping loss plot.")