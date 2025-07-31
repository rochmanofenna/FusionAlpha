import argparse
import os
import subprocess

def parse_args():
    parser = argparse.ArgumentParser(description="Run full contradiction-aware trading pipeline.")
    parser.add_argument("--csv", type=str, help="Path to CSV data file.")
    parser.add_argument("--model_path", type=str, default="./training_data/fusion_net_contradiction_weights.pth", help="Path to save/load the model.")
    parser.add_argument("--mode", type=str, choices=["prepare", "train", "predict", "evaluate"], default="train", help="Pipeline mode.")
    parser.add_argument("--target_mode", type=str, default="normalized", choices=["normalized", "binary", "rolling"], help="Target mode for training/prediction.")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs("./training_data", exist_ok=True)
    
    if args.mode == "prepare":
        subprocess.run(["python", "prepare_dataset.py", "--csv", args.csv, "--target_mode", args.target_mode])
    elif args.mode == "train":
        if not os.path.exists("./training_data/dataset.npz"):
            subprocess.run(["python", "prepare_dataset.py", "--csv", args.csv, "--target_mode", args.target_mode])
        subprocess.run(["python", "train_fusion.py", "--target_mode", args.target_mode])
    elif args.mode == "predict":
        subprocess.run(["python", "predict_fusion.py", "--target_mode", args.target_mode])
    elif args.mode == "evaluate":
        subprocess.run(["python", "evaluate_strategy.py", "--target_mode", args.target_mode])
    else:
        print("Unsupported mode.")

if __name__ == "__main__":
    main()