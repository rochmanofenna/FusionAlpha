
import numpy as np
import os

def filter_and_save_by_tag(tag):
    dataset_path = "./training_data/dataset.npz"
    predictions_path = "./training_data/predictions.npz"
    output_path = f"./training_data/{tag}_only_dataset.npz"
    
    data = np.load(dataset_path)
    preds = np.load(predictions_path, allow_pickle=True)
    tags = preds["contradiction_tags"]
    
    indices = np.where(tags == tag)[0]
    print(f"{tag} samples:", len(indices))
    
    filtered = {key: data[key][indices] for key in data}
    np.savez(output_path, **filtered)
    print(f"✅ Saved {tag} dataset to {output_path}")

if __name__ == "__main__":
    os.makedirs("training_data", exist_ok=True)
    for tag in ["underhype", "overhype", "none"]:
        filter_and_save_by_tag(tag)
