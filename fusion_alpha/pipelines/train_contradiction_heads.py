#!/usr/bin/env python3
"""
train_contradiction_heads.py

Trains a separate FusionNet head on a filtered dataset corresponding to a specific contradiction type.
Usage:
    python train_contradiction_heads.py --dataset_path ./training_data/underhype_only_dataset.npz --contradiction_type underhype --target_mode normalized --epochs 50 --batch_size 128
"""
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import joblib
from fusionnet import FusionNet

# Set device.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def compute_direction_accuracy(predictions, targets):
    pred_dir = predictions > 0
    target_dir = targets > 0
    return np.mean(pred_dir == target_dir)

def train_model(dataset_path, contradiction_type, target_mode, epochs, batch_size):
    # Load filtered dataset.
    data = np.load(dataset_path)
    tech_data = data["technical_features"]         # shape: [N, 10(+optional)]
    finbert_data = data["finbert_embeddings"]        # shape: [N, 768]
    price_data = data["price_movements"]             # shape: [N]
    sentiment_data = data["news_sentiment_scores"]     # shape: [N]
    target_returns = data["target_returns"]          # shape: [N,1]
    
    print(f"Training on dataset for contradiction type: {contradiction_type}")
    print("Total samples:", tech_data.shape[0])
    
    # Normalize technical features.
    tech_scaler = StandardScaler()
    tech_data_scaled = tech_scaler.fit_transform(tech_data)
    
    # Optionally, save the technical scaler.
    joblib.dump(tech_scaler, f"./training_data/tech_scaler_{contradiction_type}.pkl")
    print(f"Technical scaler saved to ./training_data/tech_scaler_{contradiction_type}.pkl")
    
    # Convert to torch tensors.
    tech_tensor = torch.tensor(tech_data_scaled, dtype=torch.float32).to(device)
    finbert_tensor = torch.tensor(finbert_data, dtype=torch.float32).to(device)
    price_tensor = torch.tensor(price_data, dtype=torch.float32).to(device)
    sentiment_tensor = torch.tensor(sentiment_data, dtype=torch.float32).to(device)
    target_tensor = torch.tensor(target_returns, dtype=torch.float32).to(device)
    
    # Instantiate model.
    model = FusionNet(input_dim=tech_tensor.shape[1], hidden_dim=512, use_attention=True, fusion_method='concat', target_mode=target_mode).to(device)
    
    # Choose loss.
    if target_mode == "binary":
        loss_fn = nn.BCEWithLogitsLoss()
    else:
        loss_fn = nn.MSELoss()
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    num_samples = tech_tensor.shape[0]
    num_batches = (num_samples + batch_size - 1) // batch_size
    loss_history = []
    
    model.train()
    for epoch in range(epochs):
        permutation = torch.randperm(num_samples)
        epoch_loss = 0.0
        for i in range(num_batches):
            indices = permutation[i * batch_size: (i+1) * batch_size]
            batch_tech = tech_tensor[indices]
            batch_finbert = finbert_tensor[indices]
            batch_price = price_tensor[indices]
            batch_sentiment = sentiment_tensor[indices]
            batch_target = target_tensor[indices]
            
            optimizer.zero_grad()
            # For training on filtered dataset, we assume samples already match the contradiction type.
            # So we simply use the original FinBERT embeddings.
            preds = model(batch_tech, batch_finbert).view(-1)
            loss = loss_fn(preds, batch_target.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item() * batch_tech.size(0)
        avg_loss = epoch_loss / num_samples
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    # Save the trained model.
    model_save_path = f"./training_data/fusion_{contradiction_type}_weights.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Training complete. Model saved to {model_save_path}")
    
    return model, loss_history

def main():
    parser = argparse.ArgumentParser(description="Train a FusionNet head for a specific contradiction type.")
    parser.add_argument("--dataset_path", type=str, default="./training_data/underhype_only_dataset.npz", help="Path to filtered dataset .npz file.")
    parser.add_argument("--contradiction_type", type=str, default="underhype", choices=["underhype", "overhype", "none"], help="Contradiction type to train on.")
    parser.add_argument("--target_mode", type=str, default="normalized", choices=["normalized", "binary", "rolling"], help="Target mode.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")
    args = parser.parse_args()
    
    model, loss_history = train_model(args.dataset_path, args.contradiction_type, args.target_mode, args.epochs, args.batch_size)
    
if __name__ == "__main__":
    main()