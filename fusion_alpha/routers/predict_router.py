
import numpy as np
import torch
import argparse
import joblib
from fusion_alpha.models.fusionnet import FusionNet
from fusion_alpha.pipelines.contradiction_engine import AdaptiveContradictionEngine as ContradictionEngine

def parse_args():
    parser = argparse.ArgumentParser(description="Route test samples to contradiction-specific FusionNet models.")
    parser.add_argument("--target_mode", type=str, default="normalized", choices=["normalized", "binary", "rolling"], help="Target mode used in training.")
    parser.add_argument("--contradiction_filter", type=str, default="all", choices=["underhype", "overhype", "none", "all"], help="Filter predictions by a specific contradiction type.")
    return parser.parse_args()

def main():
    args = parse_args()
    target_mode = args.target_mode
    filter_type = args.contradiction_filter

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load the full dataset.
    data = np.load("./training_data/dataset.npz")
    tech_data = data["technical_features"]         # Shape: [N, d_tech]
    finbert_data = data["finbert_embeddings"]        # Shape: [N, 768]
    price_data = data["price_movements"]             # Shape: [N]
    sentiment_data = data["news_sentiment_scores"]     # Shape: [N]
    target_returns = data["target_returns"]          # Shape: [N, 1]
    target_returns = target_returns.flatten()        # Make it 1D.

    # Convert dataset arrays to torch tensors.
    tech_tensor = torch.tensor(tech_data, dtype=torch.float32).to(device)
    finbert_tensor = torch.tensor(finbert_data, dtype=torch.float32).to(device)
    price_tensor = torch.tensor(price_data, dtype=torch.float32).to(device)
    sentiment_tensor = torch.tensor(sentiment_data, dtype=torch.float32).to(device)

    N = tech_tensor.size(0)
    print(f"Loaded {N} samples from dataset.")

    # Load all three models.
    # Assumes FusionNet was trained with fusion_method 'concat' and target_mode as specified.
    model_underhype = FusionNet(input_dim=tech_tensor.shape[1], hidden_dim=512, use_attention=True,
                                fusion_method='concat', target_mode=target_mode).to(device)
    model_overhype = FusionNet(input_dim=tech_tensor.shape[1], hidden_dim=512, use_attention=True,
                               fusion_method='concat', target_mode=target_mode).to(device)
    model_none = FusionNet(input_dim=tech_tensor.shape[1], hidden_dim=512, use_attention=True,
                           fusion_method='concat', target_mode=target_mode).to(device)
    model_underhype.load_state_dict(torch.load("./training_data/fusion_underhype_weights.pth", map_location=device))
    model_overhype.load_state_dict(torch.load("./training_data/fusion_overhype_weights.pth", map_location=device))
    model_none.load_state_dict(torch.load("./training_data/fusion_none_weights.pth", map_location=device))
    model_underhype.eval()
    model_overhype.eval()
    model_none.eval()
    print("All specialized models loaded.")

    # Load the ContradictionEngine (its role here is to determine the tag only).
    contr_engine = ContradictionEngine(embedding_dim=768).to(device)
    contr_engine.eval()

    # Prepare lists for predictions and tags.
    predictions = []
    contradiction_tags = []

    # Loop over all samples.
    with torch.no_grad():
        for i in range(N):
            tech_sample = tech_tensor[i]             # Shape: [d_tech]
            finbert_sample = finbert_tensor[i]         # Shape: [768]
            price_sample = price_tensor[i]             # Scalar tensor.
            sentiment_sample = sentiment_tensor[i]     # Scalar tensor.
            
            # Determine contradiction tag using ContradictionEngine.
            # We ignore the updated embedding since each head expects raw FinBERT.
            _, ctype = contr_engine(finbert_sample, tech_sample, price_sample, sentiment_sample)
            if ctype is None:
                ctype = "none"
            contradiction_tags.append(ctype)
            
            # Route the sample based on its contradiction tag.
            # Each head expects the raw FinBERT embedding.
            tech_input = tech_sample.unsqueeze(0)       # Shape: [1, d_tech]
            finbert_input = finbert_sample.unsqueeze(0)   # Shape: [1, 768]
            if ctype == "underhype":
                pred = model_underhype(tech_input, finbert_input)
            elif ctype == "overhype":
                pred = model_overhype(tech_input, finbert_input)
            else:
                pred = model_none(tech_input, finbert_input)
            # TODO: Avoid .cpu().numpy() in live router for better latency
            predictions.append(pred.detach().flatten()[0].item())
    
    predictions = np.array(predictions)
    contradiction_tags = np.array(contradiction_tags)

    # If filtering is requested, zero out predictions for samples not matching the filter.
    if filter_type != "all":
        mask = (contradiction_tags == filter_type)
        num_filtered = np.sum(~mask)
        predictions[~mask] = 0.0
        print(f"Filtering applied: {num_filtered} samples zeroed out (not '{filter_type}').")
    
    # Save final predictions.
    output_path = "./training_data/predictions_routed.npz"
    np.savez(output_path,
             predictions=predictions,
             target_returns=target_returns,
             contradiction_tags=contradiction_tags)
    print("Predictions saved to", output_path)

if __name__ == "__main__":
    args = parse_args()
    main()
