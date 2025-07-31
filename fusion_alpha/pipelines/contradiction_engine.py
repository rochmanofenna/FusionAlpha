import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveContradictionEngine(nn.Module):
    def __init__(self, embedding_dim=768):
        """
        Initializes learnable thresholds for contradiction detection.
        """
        super(AdaptiveContradictionEngine, self).__init__()
        self.embedding_dim = embedding_dim

        # Initialize thresholds as trainable parameters.
        self.pos_sent_thresh = nn.Parameter(torch.tensor(0.1))
        self.neg_sent_thresh = nn.Parameter(torch.tensor(-0.1))
        self.drop_thresh = nn.Parameter(torch.tensor(0.01))
        self.rise_thresh = nn.Parameter(torch.tensor(0.01))
        
        # A learned nonlinear transformation to generate updated semantic embeddings.
        self.transform = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.Tanh()
        )

    def forward(self, finbert_embedding, technical_features, price_movement, news_sentiment_score):
        # Convert sentiment and movement to scalars if needed.
        sentiment = news_sentiment_score.item() if isinstance(news_sentiment_score, torch.Tensor) and news_sentiment_score.dim() == 0 else news_sentiment_score
        movement = price_movement.item() if isinstance(price_movement, torch.Tensor) and price_movement.dim() == 0 else price_movement
        
        # Debug: print inputs and thresholds.
        print("ContradictionEngine Debug:")
        print(f"  Sentiment: {sentiment}, Price Movement: {movement}")
        print(f"  Thresholds -> pos: {self.pos_sent_thresh.item()}, neg: {self.neg_sent_thresh.item()}, drop: {self.drop_thresh.item()}, rise: {self.rise_thresh.item()}")
        
        contradiction_detected = False
        contradiction_type = None

        if sentiment > self.pos_sent_thresh and movement < -self.drop_thresh:
            contradiction_detected = True
            contradiction_type = "overhype"
        elif sentiment < self.neg_sent_thresh and movement > self.rise_thresh:
            contradiction_detected = True
            contradiction_type = "underhype"
        
        if contradiction_detected:
            print(f"  Contradiction detected: {contradiction_type}")
            updated_embedding = self.transform(finbert_embedding)
            return updated_embedding, contradiction_type
        else:
            print("  No contradiction detected.")
            return finbert_embedding, None

# Alias for backward compatibility
ContradictionEngine = AdaptiveContradictionEngine

if __name__ == "__main__":
    engine = ContradictionEngine()
    dummy_embedding = torch.randn(768)
    dummy_tech = torch.randn(10)
    price_movement = torch.tensor(-0.05)
    news_sentiment_score = torch.tensor(0.8)
    
    updated_emb, ctype = engine(dummy_embedding, dummy_tech, price_movement, news_sentiment_score)
    print("Contradiction type:", ctype)
    print("First 5 elements of updated embedding:", updated_emb[:5])