import numpy as np
import matplotlib.pyplot as plt
import argparse
from sklearn.metrics import accuracy_score

def compute_direction_accuracy(predictions, targets):
    pred_dir = predictions > 0
    target_dir = targets > 0
    return np.mean(pred_dir == target_dir)

def compute_sharpe_ratio(returns, risk_free_rate=0.0):
    excess = returns - risk_free_rate
    return excess.mean() / (excess.std() + 1e-8)

def evaluate_strategy(predictions, targets, contradiction_tags, target_mode):
    overall_dir_acc = compute_direction_accuracy(predictions, targets)
    overall_avg_return = np.mean(predictions)
    overall_sharpe = compute_sharpe_ratio(predictions)

    print("evaluate_strategy.py Debug:")
    print("  Overall baseline average target return:", np.mean(targets))

    unique_tags, counts = np.unique(contradiction_tags, return_counts=True)
    print("  Contradiction Tags Distribution:", dict(zip(unique_tags, counts)))

    per_type_metrics = {}
    for tag in unique_tags:
        mask = contradiction_tags == tag
        if np.sum(mask) > 0:
            if target_mode == "binary":
                binary_preds = predictions[mask] > 0.5
                binary_targets = targets[mask] == 1
                acc = accuracy_score(binary_targets, binary_preds)
            else:
                acc = compute_direction_accuracy(predictions[mask], targets[mask])
            avg_ret = np.mean(predictions[mask])
            sharpe = compute_sharpe_ratio(predictions[mask])
            per_type_metrics[tag] = (acc, avg_ret, sharpe)
            print(f"  Metrics for {tag}: Direction Accuracy = {acc:.2%}, Average Return = {avg_ret:.4f}, Sharpe Ratio = {sharpe:.4f}")

    return (overall_dir_acc, overall_avg_return, overall_sharpe), per_type_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_mode", type=str, default="normalized", choices=["normalized", "binary", "rolling"])
    parser.add_argument("--filter_underhype", action="store_true")
    parser.add_argument("--filter_overhype", action="store_true")
    parser.add_argument("--filter_none", action="store_true")
    args = parser.parse_args()

    target_mode = args.target_mode

    data = np.load("./training_data/predictions_routed.npz", allow_pickle=True)
    predictions = data["predictions"]
    targets = data["target_returns"]
    contradiction_tags = data["contradiction_tags"]

    # Conditional filtering
    if args.filter_underhype or args.filter_overhype or args.filter_none:
        if args.filter_underhype:
            filter_type = "underhype"
        elif args.filter_overhype:
            filter_type = "overhype"
        else:
            filter_type = "none"
        mask = contradiction_tags == filter_type
        predictions = predictions[mask]
        targets = targets[mask]
        contradiction_tags = contradiction_tags[mask]
        print(f"Evaluation filtering applied: {np.sum(~mask)} samples removed.")

    overall, per_type = evaluate_strategy(predictions, targets, contradiction_tags, target_mode)
    print("Overall Metrics:")
    print("  Direction Accuracy: {:.2%}".format(overall[0]))
    print("  Average Predicted Return: {:.4f}".format(overall[1]))
    print("  Sharpe Ratio: {:.4f}".format(overall[2]))

    plt.figure(figsize=(6, 4))
    unique, counts = np.unique(contradiction_tags, return_counts=True)
    plt.bar(unique, counts)
    plt.xlabel("Contradiction Type")
    plt.ylabel("Count")
    plt.title("Contradiction Type Distribution")
    plt.show()