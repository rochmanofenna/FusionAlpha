import numpy as np
import torch
import joblib
import argparse
from fusionnet import FusionNet
from contradiction_engine import ContradictionEngine

parser = argparse.ArgumentParser()
parser.add_argument("--target_mode", type=str, default="normalized", choices=["normalized", "binary", "rolling"])
parser.add_argument("--filter_underhype", action="store_true", help="Filter predictions to only include 'underhype' samples.")
args = parser.parse_args()
target_mode = args.target_mode
filter_underhype = args.filter_underhype

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = FusionNet(input_dim=10, hidden_dim=512, use_attention=True, fusion_method='concat', target_mode=target_mode).to(device)
model.load_state_dict(torch.load("./training_data/fusion_underhype_weights_fold5.pth", map_location=device))
model.eval()
contradiction_engine = ContradictionEngine(embedding_dim=768).to(device)

data = np.load("./training_data/dataset.npz")
tech_data = data["technical_features"]
finbert_data = data["finbert_embeddings"]
price_data = data["price_movements"]
sentiment_data = data["news_sentiment_scores"]

tech_tensor = torch.tensor(tech_data, dtype=torch.float32).to(device)
finbert_tensor = torch.tensor(finbert_data, dtype=torch.float32).to(device)
price_tensor = torch.tensor(price_data, dtype=torch.float32).to(device)
sentiment_tensor = torch.tensor(sentiment_data, dtype=torch.float32).to(device)

updated_embeddings = []
contradiction_tags = []
for i in range(finbert_tensor.size(0)):
    updated_emb, ctype = contradiction_engine(finbert_tensor[i], tech_tensor[i], price_tensor[i], sentiment_tensor[i])
    updated_embeddings.append(updated_emb)
    contradiction_tags.append(ctype if ctype is not None else "none")
updated_embeddings = torch.stack(updated_embeddings)

with torch.no_grad():
    predictions = model(tech_tensor, updated_embeddings)
predictions = predictions.cpu().numpy().flatten()

if target_mode != "binary":
    scaler = joblib.load("./training_data/target_scaler.pkl")
    predictions_final = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
else:
    predictions_final = predictions

if filter_underhype:
    # Zero out predictions for samples not labeled "underhype"
    tags = np.array(contradiction_tags)
    mask = tags == "underhype"
    num_filtered = np.sum(~mask)
    predictions_final[~mask] = 0.0
    print(f"Filtering applied: {num_filtered} samples zeroed out (not underhype).")

print("Predictions stats:")
print(f"  Before transform: min {predictions.min()}, max {predictions.max()}, mean {predictions.mean()}")
print(f"  Final predictions: min {predictions_final.min()}, max {predictions_final.max()}, mean {predictions_final.mean()}")

np.savez("./training_data/predictions.npz",
         predictions=predictions_final,
         target_returns=data["target_returns"],
         contradiction_tags=np.array(contradiction_tags))
print("Predictions saved to ./training_data/predictions.npz")