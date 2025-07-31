#!/usr/bin/env python3
"""
live_router.py

Live signal generator that:
  - Accepts technical features, FinBERT embedding, price movement, sentiment score, and current price.
  - Uses the ContradictionEngine to tag the sample.
  - Routes the sample to the appropriate specialized FusionNet model.
  - Outputs the predicted return and calculates the projected target price.
Usage:
  python live_router.py --tech "[0.1,0.2,...,0.3]" --finbert "[0.1,0.2,...,0.5]" --price 0.012 --sentiment 0.7 --current_price 150 --target_mode normalized
"""
import argparse
import ast
import numpy as np
import torch
from fusionnet import FusionNet
from contradiction_engine import ContradictionEngine

def parse_args():
    parser = argparse.ArgumentParser(description="Live router for contradiction-aware trading signal generation.")
    parser.add_argument("--tech", type=str, required=True, help="Technical features as a JSON list string (e.g., '[0.1,0.2,...]')")
    parser.add_argument("--finbert", type=str, required=True, help="FinBERT embedding as a JSON list string (length 768)")
    parser.add_argument("--price", type=float, required=True, help="Price movement (e.g., 0.01 for 1% increase)")
    parser.add_argument("--sentiment", type=float, required=True, help="News sentiment score (e.g., 0.7)")
    parser.add_argument("--current_price", type=float, required=True, help="Current asset price")
    parser.add_argument("--target_mode", type=str, default="normalized", choices=["normalized", "binary", "rolling"], help="Target mode used in training")
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    # Parse inputs.
    tech_features = torch.tensor(ast.literal_eval(args.tech), dtype=torch.float32).to(device)
    finbert_embedding = torch.tensor(ast.literal_eval(args.finbert), dtype=torch.float32).to(device)
    price_movement = torch.tensor(args.price, dtype=torch.float32).to(device)
    sentiment_score = torch.tensor(args.sentiment, dtype=torch.float32).to(device)
    current_price = args.current_price

    # Load ContradictionEngine.
    contr_engine = ContradictionEngine(embedding_dim=768).to(device)
    contr_engine.eval()
    updated_emb, ctype = contr_engine(finbert_embedding, tech_features, price_movement, sentiment_score)
    if ctype is None:
        ctype = "none"
    print("Contradiction tag determined:", ctype)
    
    # Load specialized models.
    model_underhype = FusionNet(input_dim=len(tech_features), hidden_dim=512, use_attention=True, fusion_method='concat', target_mode=args.target_mode).to(device)
    model_overhype = FusionNet(input_dim=len(tech_features), hidden_dim=512, use_attention=True, fusion_method='concat', target_mode=args.target_mode).to(device)
    model_none = FusionNet(input_dim=len(tech_features), hidden_dim=512, use_attention=True, fusion_method='concat', target_mode=args.target_mode).to(device)
    model_underhype.load_state_dict(torch.load("./training_data/fusion_underhype_weights.pth", map_location=device))
    model_overhype.load_state_dict(torch.load("./training_data/fusion_overhype_weights.pth", map_location=device))
    model_none.load_state_dict(torch.load("./training_data/fusion_none_weights.pth", map_location=device))
    model_underhype.eval()
    model_overhype.eval()
    model_none.eval()
    
    # Route sample.
    tech_input = tech_features.unsqueeze(0)     # [1, d_tech]
    finbert_input = finbert_embedding.unsqueeze(0) # [1, 768]
    if ctype == "underhype":
        prediction = model_underhype(tech_input, finbert_input)
    elif ctype == "overhype":
        prediction = model_overhype(tech_input, finbert_input)
    else:
        prediction = model_none(tech_input, finbert_input)
    
    prediction = prediction.item()
    # Calculate projected price.
    target_price = current_price * (1 + prediction)
    
    # Determine action.
    threshold = 0.01
    if ctype == "underhype" and prediction > threshold:
        action = "LONG"
    elif ctype == "overhype" and prediction < -threshold:
        action = "SHORT"
    else:
        action = "NO_ACTION"
    
    print("Live Signal Output:")
    print("  Contradiction Type:", ctype)
    print("  Predicted Return:", prediction)
    print("  Projected Target Price:", target_price)
    print("  Action:", action)
    print("  Confidence (abs(prediction)):", abs(prediction))
    
if __name__ == "__main__":
    main()