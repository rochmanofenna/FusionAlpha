import optuna
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from fusionnet import FusionNet

# Set device for GPU acceleration.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

###############################################
# Preprocessing Block
###############################################

# Load CSV and inspect columns.
data = pd.read_csv("training_data/combined_data_with_target.csv")
print("Initial data shape:", data.shape)
print("Columns in CSV:", data.columns.tolist())

# Define feature columns.
feature_columns = [
    "Close", "Open", "High", "Low", "Volume", "SMA20", 
    "SMA50", "EMA20", "EMA50", "RSI14", "MACD", "StochK", 
    "StochD", "HistoricalVol20", "ATR14", "ImpliedVol"
]

# Replace infinities with NaN.
data.replace([np.inf, -np.inf], np.nan, inplace=True)

# Print number of NaNs per feature column.
print("Missing values per feature column before cleaning:")
print(data[feature_columns].isna().sum())

# Only drop rows where the target is missing.
data.dropna(subset=['next_return'], inplace=True)
print("Data shape after dropping rows with missing target:", data.shape)

# Fill missing feature values (using forward-fill; adjust method as needed).
data[feature_columns] = data[feature_columns].fillna(method='ffill')
print("Missing values per feature column after forward-fill:")
print(data[feature_columns].isna().sum())

# Extract features and target.
X_raw = data[feature_columns].values
y = data['next_return'].values

# Debug: Print shapes to confirm data isn't empty.
print("Shape of raw features (X_raw):", X_raw.shape)
print("Shape of target (y):", y.shape)

# Normalize features.
scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

# Confirm no NaNs are present after scaling.
if np.isnan(X).any():
    print("Warning: NaNs found in normalized features!")
if np.isnan(y).any():
    print("Warning: NaNs found in target!")

print("Shape of normalized features (X):", X.shape)

###############################################
# End of Preprocessing Block
###############################################

# Split data into training (80%) and validation (20%).
split_idx = int(len(X) * 0.8)
X_train, y_train = X[:split_idx], y[:split_idx]
X_val, y_val = X[split_idx:], y[split_idx:]

# Create an Optuna study with SQLite storage.
study = optuna.create_study(
    direction="minimize",
    storage="sqlite:///optuna_study.db",
    load_if_exists=True,
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=2)
)

def objective(trial):
    hidden_dim = trial.suggest_int('hidden_dim', 32, 128)
    use_attention = trial.suggest_categorical('use_attention', [False, True])
    fusion_method = trial.suggest_categorical('fusion_method', ['concat', 'average'])
    lr = trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True)
    
    input_dim = X_train.shape[1]
    model = FusionNet(input_dim, hidden_dim=hidden_dim, use_attention=use_attention, fusion_method=fusion_method).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    model.train()
    
    batch_size = 128
    num_samples = X_train.shape[0]
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    for epoch in range(5):
        epoch_loss = 0.0
        permutation = np.random.permutation(num_samples)
        X_train_shuffled = X_train[permutation]
        y_train_shuffled = y_train[permutation]
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i+1) * batch_size, num_samples)
            batch_X = torch.from_numpy(X_train_shuffled[start_idx:end_idx].astype(np.float32)).to(device)
            batch_y = torch.from_numpy(y_train_shuffled[start_idx:end_idx].astype(np.float32)).to(device)
            optimizer.zero_grad()
            pred = model(batch_X, batch_X)
            pred = pred.view(-1)
            loss = loss_fn(pred, batch_y)
            if torch.isnan(loss):
                print(f"NaN loss detected at epoch {epoch+1}, batch {i+1}. Pruning trial.")
                raise optuna.TrialPruned("Loss became NaN")
            with torch.autograd.set_detect_anomaly(True):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            print(f"[Trial {trial.number}] Epoch {epoch+1}, Batch {i+1}/{num_batches}, Loss: {loss.item():.4f}")
            epoch_loss += loss.item() * (end_idx - start_idx)
        epoch_loss /= num_samples
        trial.report(epoch_loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    model.eval()
    with torch.no_grad():
        Xv_tensor = torch.from_numpy(X_val.astype(np.float32)).to(device)
        yv_tensor = torch.from_numpy(y_val.astype(np.float32)).to(device)
        pred_val = model(Xv_tensor, Xv_tensor).view(-1)
        val_loss = loss_fn(pred_val, yv_tensor).item()
    return val_loss

study.optimize(objective, n_trials=50, show_progress_bar=True)
best_params = study.best_params
print("Optuna best parameters:", best_params)

# Train a final model on the combined train+val dataset.
batch_size = 128
X_all = np.concatenate([X_train, X_val], axis=0)
y_all = np.concatenate([y_train, y_val], axis=0)
num_samples_all = X_all.shape[0]
num_batches_all = (num_samples_all + batch_size - 1) // batch_size

best_model = FusionNet(input_dim, hidden_dim=best_params['hidden_dim'],
                       use_attention=best_params['use_attention'],
                       fusion_method=best_params['fusion_method']).to(device)
best_model.train()
optimizer = optim.Adam(best_model.parameters(), lr=best_params['learning_rate'])
loss_fn = nn.MSELoss()

for epoch in range(5):
    epoch_loss = 0.0
    permutation = np.random.permutation(num_samples_all)
    X_all_shuffled = X_all[permutation]
    y_all_shuffled = y_all[permutation]
    for i in range(num_batches_all):
        start_idx = i * batch_size
        end_idx = min((i+1) * batch_size, num_samples_all)
        batch_X = torch.from_numpy(X_all_shuffled[start_idx:end_idx].astype(np.float32)).to(device)
        batch_y = torch.from_numpy(y_all_shuffled[start_idx:end_idx].astype(np.float32)).to(device)
        optimizer.zero_grad()
        pred_all = best_model(batch_X, batch_X).view(-1)
        loss = loss_fn(pred_all, batch_y)
        with torch.autograd.set_detect_anomaly(True):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(best_model.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_loss += loss.item() * (end_idx - start_idx)
    epoch_loss /= num_samples_all
    print(f"Final training epoch {epoch+1}/5, Loss: {epoch_loss:.4f}")
        
best_model.eval()
best_model.save_model("fusion_net_best_weights.pth")
print("Best model saved as fusion_net_best_weights.pth")