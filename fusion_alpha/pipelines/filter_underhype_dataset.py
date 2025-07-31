import numpy as np
import os

def filter_underhype_dataset(dataset_path, predictions_path, output_path):
    """
    Loads the full dataset and predictions, filters samples with contradiction_tag "underhype",
    and saves the filtered dataset.
    """
    data = np.load(dataset_path)
    preds = np.load(predictions_path, allow_pickle=True)
    tags = preds["contradiction_tags"]
    
    underhype_indices = np.where(tags == "underhype")[0]
    print("Number of underhype samples:", len(underhype_indices))
    
    filtered_data = {}
    for key in data.keys():
        filtered_data[key] = data[key][underhype_indices]
    
    np.savez(output_path, **filtered_data)
    print("Filtered dataset saved to", output_path)

if __name__ == "__main__":
    dataset_path = "./training_data/dataset.npz"
    predictions_path = "./training_data/predictions.npz"
    output_path = "./training_data/underhype_only_dataset.npz"
    os.makedirs("./training_data", exist_ok=True)
    filter_underhype_dataset(dataset_path, predictions_path, output_path)