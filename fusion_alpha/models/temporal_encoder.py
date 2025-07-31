# temporal_encoder.py
import torch
import torch.nn as nn

class TransformerTimeSeriesEncoder(nn.Module):
    def __init__(self, input_dim, model_dim, num_layers=2, nhead=4, dropout=0.1):
        super(TransformerTimeSeriesEncoder, self).__init__()
        self.input_fc = nn.Linear(input_dim, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_fc = nn.Linear(model_dim, model_dim)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch, seq_len, input_dim]
        Returns:
            encoded: Tensor of shape [batch, model_dim] (e.g. mean-pooled over sequence)
        """
        x = self.input_fc(x)
        x = self.transformer_encoder(x)
        x = self.output_fc(x)
        # Mean pool over sequence dimension.
        encoded = x.mean(dim=1)
        return encoded

if __name__ == "__main__":
    # Quick test.
    batch_size = 8
    seq_len = 10
    input_dim = 20
    model_dim = 32
    dummy_input = torch.randn(batch_size, seq_len, input_dim)
    encoder = TransformerTimeSeriesEncoder(input_dim, model_dim)
    output = encoder(dummy_input)
    print("Transformer encoder output shape:", output.shape)