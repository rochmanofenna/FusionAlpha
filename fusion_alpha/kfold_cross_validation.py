# kfold_cross_validation.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from fusionnet import FusionNet
from contradiction_engine import AdaptiveContradictionEngine
from sklearn.model_selection import KFold

def compute_direction_accuracy(predictions, targets):
    pred_dir = predictions > 0
    target_dir = targets > 0
    return (pred_dir == target_dir).mean()

def compute_sharpe_ratio(returns, risk_free_rate=0.0):
    excess = returns - risk_free_rate
    return excess.mean() / (excess.std() + 1e-8)

def run_kfold_cv(dataset_path, n_splits=5, num_epochs=5):
    data = np.load(dataset_path)
    tech_data = data["technical_features"]
    finbert_data = data["finbert_embeddings"]
    price_data = data["price_movements"]
    sentiment_data = data["news_sentiment_scores"]
    target_returns = data["target_returns"]
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    metrics = []
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for train_index, test_index in kf.split(tech_data):
        # Prepare train/test tensors.
        tech_train = torch.tensor(tech_data[train_index], dtype=torch.float32).to(device)
        finbert_train = torch.tensor(finbert_data[train_index], dtype=torch.float32).to(device)
        price_train = torch.tensor(price_data[train_index], dtype=torch.float32).to(device)
        sentiment_train = torch.tensor(sentiment_data[train_index], dtype=torch.float32).to(device)
        target_train = torch.tensor(target_returns[train_index], dtype=torch.float32).to(device)
        
        tech_test = torch.tensor(tech_data[test_index], dtype=torch.float32).to(device)
        finbert_test = torch.tensor(finbert_data[test_index], dtype=torch.float32).to(device)
        price_test = torch.tensor(price_data[test_index], dtype=torch.float32).to(device)
        sentiment_test = torch.tensor(sentiment_data[test_index], dtype=torch.float32).to(device)
        target_test = torch.tensor(target_returns[test_index], dtype=torch.float32).to(device)
        
        # Initialize model and contradiction engine.
        model = FusionNet(tech_input_dim=10, hidden_dim=512, use_attention=True, fusion_method='concat').to(device)
        contradiction_engine = AdaptiveContradictionEngine(embedding_dim=768).to(device)
        loss_fn = nn.MSELoss()
        optimizer = optim.Adam(list(model.parameters()) + list(contradiction_engine.parameters()), lr=1e-3)
        
        # Simple training loop.
        for epoch in range(num_epochs):
            model.train()
            permutation = torch.randperm(tech_train.size(0))
            epoch_loss = 0.0
            for i in range(0, tech_train.size(0), 128):
                indices = permutation[i:i+128]
                batch_tech = tech_train[indices]
                batch_finbert = finbert_train[indices]
                batch_price = price_train[indices]
                batch_sentiment = sentiment_train[indices]
                batch_target = target_train[indices]
                
                optimizer.zero_grad()
                updated_embeddings = []
                for j in range(batch_finbert.size(0)):
                    updated_emb, _ = contradiction_engine(batch_finbert[j], batch_tech[j], batch_price[j], batch_sentiment[j])
                    updated_embeddings.append(updated_emb)
                updated_embeddings = torch.stack(updated_embeddings)
                pred = model(batch_tech, updated_embeddings).view(-1)
                loss = loss_fn(pred, batch_target.view(-1))
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch_tech.size(0)
            #print(f"Fold training epoch {epoch+1}: Loss = {epoch_loss/tech_train.size(0):.4f}")
        
        # Evaluation on test fold.
        model.eval()
        with torch.no_grad():
            updated_embeddings_test = []
            for j in range(finbert_test.size(0)):
                updated_emb, _ = contradiction_engine(finbert_test[j], tech_test[j], price_test[j], sentiment_test[j])
                updated_embeddings_test.append(updated_emb)
            updated_embeddings_test = torch.stack(updated_embeddings_test)
            predictions = model(tech_test, updated_embeddings_test).view(-1)
            predictions_np = predictions.cpu().numpy()
            targets_np = target_test.view(-1).cpu().numpy()
            dir_acc = compute_direction_accuracy(predictions_np, targets_np)
            avg_ret = predictions_np.mean()
            sharpe = compute_sharpe_ratio(predictions_np)
            metrics.append((dir_acc, avg_ret, sharpe))
            print(f"Fold metrics: Direction Acc: {dir_acc:.2%}, Avg Return: {avg_ret:.4f}, Sharpe: {sharpe:.4f}")
    return metrics

if __name__ == "__main__":
    metrics = run_kfold_cv("./training_data/dataset.npz", n_splits=5, num_epochs=5)
    metrics = np.array(metrics)
    print("Average metrics across folds:")
    print("Direction Accuracy: {:.2%}".format(metrics[:,0].mean()))
    print("Average Return: {:.4f}".format(metrics[:,1].mean()))
    print("Sharpe Ratio: {:.4f}".format(metrics[:,2].mean()))