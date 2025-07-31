# prepare_dataset.py
import numpy as np
import pandas as pd
import os
import argparse
from sklearn.preprocessing import StandardScaler
import joblib

def prepare_dataset(csv_path, output_npz_path, scaler_path, target_mode="normalized"):
    df = pd.read_csv(csv_path)
    
    # Extract features
    technical_cols = [f"tech{i}" for i in range(1, 11)]
    finbert_cols = [f"finbert_{i}" for i in range(768)]
    price_col = "price_movement"
    sentiment_col = "news_sentiment_score"

    required_cols = technical_cols + finbert_cols + [price_col, sentiment_col, "next_return"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    # Filter valid rows
    df = df.dropna(subset=required_cols)

    # Scale technical features
    scaler = StandardScaler()
    tech_scaled = scaler.fit_transform(df[technical_cols])
    joblib.dump(scaler, scaler_path)
    print("Technical features normalized.")

    # Prepare arrays
    finbert_embeddings = df[finbert_cols].values.astype(np.float32)
    price_movements = df[price_col].values.astype(np.float32)
    sentiment_scores = df[sentiment_col].values.astype(np.float32)

    if target_mode == "binary":
        target_returns = (df["next_return"] > 0).astype(np.float32)
    else:
        target_returns = df["next_return"].values.astype(np.float32)
        if target_mode == "normalized":
            target_scaler = StandardScaler()
            target_returns = target_scaler.fit_transform(target_returns.reshape(-1, 1)).flatten()
            joblib.dump(target_scaler, "./training_data/target_scaler.pkl")

    # Save as .npz
    np.savez(output_npz_path,
             technical_features=tech_scaled,
             finbert_embeddings=finbert_embeddings,
             price_movements=price_movements,
             news_sentiment_scores=sentiment_scores,
             target_returns=target_returns)
    
    print(f"Dataset saved to {output_npz_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_mode", type=str, default="normalized", choices=["normalized", "binary", "rolling"])
    args = parser.parse_args()

    csv_path = "./training_data/combined_data_with_target_with_real_finbert.csv"
    output_npz_path = "./training_data/dataset.npz"
    scaler_path = "./training_data/technical_scaler.pkl"

    prepare_dataset(csv_path, output_npz_path, scaler_path, target_mode=args.target_mode)