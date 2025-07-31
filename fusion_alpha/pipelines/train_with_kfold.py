#!/usr/bin/env python3
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from fusionnet import FusionNet
from contradiction_engine import ContradictionEngine
# (Assume other modules like StandardScaler are imported as needed)

def compute_direction_accuracy(predictions, targets):
    pred_dir = predictions > 0
    target_dir = targets > 0
    return np.mean(pred_dir == target_dir)

def compute_sharpe_ratio(returns, risk_free_rate=0.0):
    excess = returns - risk_free_rate
    return excess.mean() / (excess.std() + 1e-8)

def run_kfold_cv(dataset_path, n_splits, num_epochs, batch_size, target_mode, save_models=False):
    data = np.load(dataset_path)
    tech_data = data["technical_features"]
    finbert_data = data["finbert_embeddings"]
    price_data = data["price_movements"]
    sentiment_data = data["news_sentiment_scores"]
    target_returns = data["target_returns"]
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    metrics = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    fold = 1
    for train_idx, val_idx in kf.split(tech_data):
        print(f"Starting Fold {fold}")
        # Prepare training tensors.
        tech_train = torch.tensor(tech_data[train_idx], dtype=torch.float32).to(device)
        finbert_train = torch.tensor(finbert_data[train_idx], dtype=torch.float32).to(device)
        price_train = torch.tensor(price_data[train_idx], dtype=torch.float32).to(device)
        sentiment_train = torch.tensor(sentiment_data[train_idx], dtype=torch.float32).to(device)
        target_train = torch.tensor(target_returns[train_idx], dtype=torch.float32).to(device)
        
        tech_val = torch.tensor(tech_data[val_idx], dtype=torch.float32).to(device)
        finbert_val = torch.tensor(finbert_data[val_idx], dtype=torch.float32).to(device)
        price_val = torch.tensor(price_data[val_idx], dtype=torch.float32).to(device)
        sentiment_val = torch.tensor(sentiment_data[val_idx], dtype=torch.float32).to(device)
        target_val = torch.tensor(target_returns[val_idx], dtype=torch.float32).to(device)
        
        model = FusionNet(input_dim=tech_train.shape[1], hidden_dim=512, use_attention=True, fusion_method='concat', target_mode=target_mode).to(device)
        contr_engine = ContradictionEngine(embedding_dim=768).to(device)
        if target_mode == "binary":
            loss_fn = nn.BCEWithLogitsLoss()
        else:
            loss_fn = nn.MSELoss()
        optimizer = optim.Adam(list(model.parameters()) + list(contr_engine.parameters()), lr=1e-3)
        
        n_train = tech_train.shape[0]
        num_batches = (n_train + batch_size - 1) // batch_size
        
        for epoch in range(num_epochs):
            model.train()
            contr_engine.train()
            permutation = torch.randperm(n_train)
            epoch_loss = 0.0
            for i in range(num_batches):
                indices = permutation[i*batch_size:(i+1)*batch_size]
                batch_tech = tech_train[indices]
                batch_finbert = finbert_train[indices]
                batch_price = price_train[indices]
                batch_sentiment = sentiment_train[indices]
                batch_target = target_train[indices]
                
                optimizer.zero_grad()
                updated_embeddings = []
                for j in range(batch_finbert.size(0)):
                    upd_emb, _ = contr_engine(batch_finbert[j], batch_tech[j], batch_price[j], batch_sentiment[j])
                    updated_embeddings.append(upd_emb)
                updated_embeddings = torch.stack(updated_embeddings)
                preds = model(batch_tech, updated_embeddings).view(-1)
                loss = loss_fn(preds, batch_target.view(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item() * batch_finbert.size(0)
            avg_loss = epoch_loss / n_train
            print(f"[Fold {fold}] Epoch {epoch+1}/{num_epochs} Loss: {avg_loss:.4f}")
        
        model.eval()
        contr_engine.eval()
        with torch.no_grad():
            updated_val_embeddings = []
            for j in range(finbert_val.size(0)):
                upd_emb, _ = contr_engine(finbert_val[j], tech_val[j], price_val[j], sentiment_val[j])
                updated_val_embeddings.append(upd_emb)
            updated_val_embeddings = torch.stack(updated_val_embeddings)
            val_preds = model(tech_val, updated_val_embeddings).view(-1).cpu().numpy()
            val_targets = target_val.view(-1).cpu().numpy()
        dir_acc = compute_direction_accuracy(val_preds, val_targets)
        avg_ret = val_preds.mean()
        sharpe = compute_sharpe_ratio(val_preds)
        metrics.append({"direction_accuracy": dir_acc, "average_return": avg_ret, "sharpe_ratio": sharpe})
        print(f"[Fold {fold}] Metrics: Direction Accuracy: {dir_acc:.2%}, Avg Return: {avg_ret:.4f}, Sharpe: {sharpe:.4f}")
        # Optionally save the model.
        if save_models:
            torch.save(model.state_dict(), f"./training_data/fusion_underhype_weights_fold{fold}.pth")
            print(f"Model for fold {fold} saved.")
        fold += 1
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Train contradiction-aware model with 5-fold CV.")
    parser.add_argument("--dataset_path", type=str, default="./training_data/dataset.npz", help="Path to dataset .npz file.")
    parser.add_argument("--target_mode", type=str, default="normalized", choices=["normalized", "binary", "rolling"])
    parser.add_argument("--epochs", type=int, default=75, help="Epochs per fold.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")
    parser.add_argument("--n_splits", type=int, default=5, help="Number of folds.")
    parser.add_argument("--save_models", action="store_true", help="Save model for each fold.")
    args = parser.parse_args()
    
    metrics = run_kfold_cv(args.dataset_path, args.n_splits, args.epochs, args.batch_size, args.target_mode, args.save_models)
    avg_dir = np.mean([m["direction_accuracy"] for m in metrics])
    avg_ret = np.mean([m["average_return"] for m in metrics])
    avg_sharpe = np.mean([m["sharpe_ratio"] for m in metrics])
    print("Average Metrics Across Folds:")
    print(f"  Direction Accuracy: {avg_dir:.2%}")
    print(f"  Average Return: {avg_ret:.4f}")
    print(f"  Sharpe Ratio: {avg_sharpe:.4f}")

if __name__ == "__main__":
    main()