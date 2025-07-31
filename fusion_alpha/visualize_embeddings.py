# visualize_embeddings.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def visualize_embeddings(original, modified, method="pca"):
    """
    Visualizes original vs. modified embeddings using PCA or t-SNE.
    Args:
        original: numpy array of shape [n, d]
        modified: numpy array of shape [n, d]
        method: "pca" or "tsne"
    """
    if method == "pca":
        reducer = PCA(n_components=2)
    elif method == "tsne":
        reducer = TSNE(n_components=2, perplexity=30, n_iter=300)
    else:
        raise ValueError("Unsupported method: choose 'pca' or 'tsne'")
    
    orig_2d = reducer.fit_transform(original)
    mod_2d = reducer.fit_transform(modified)
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(orig_2d[:, 0], orig_2d[:, 1], c="blue", alpha=0.6)
    plt.title("Original FinBERT Embeddings")
    
    plt.subplot(1, 2, 2)
    plt.scatter(mod_2d[:, 0], mod_2d[:, 1], c="red", alpha=0.6)
    plt.title("Contradiction-Modified Embeddings")
    plt.show()

if __name__ == "__main__":
    # Load saved embeddings from a .npz file.
    data = np.load("./training_data/dataset.npz")
    original = data["finbert_embeddings"]
    
    # For demonstration, we simulate modified embeddings by a simple transformation.
    modified = np.tanh(original)  # In practice, use your contradiction engine output.
    
    visualize_embeddings(original, modified, method="pca")