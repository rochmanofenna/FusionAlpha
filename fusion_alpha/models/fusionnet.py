import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def init_weights(m):
    """Initialize weights for linear layers using Xavier uniform initialization."""
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class FusionNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, output_dim=1, use_attention=False, fusion_method='average', target_mode="normalized", dropout_prob=0.5, logit_clamp_min=-10, logit_clamp_max=10):
        """
        fusion_method: 'average' or 'concat'
            - For 'concat', fc1 input dimension becomes input_dim + 768.
            - For 'average', we simply average emb1 and emb2 and fc1 input is input_dim.
        target_mode: "normalized", "binary", or "rolling". Determines loss function and output activation.
            - For "binary", the model outputs raw logits during training and applies clamped sigmoid during evaluation.
            - For "normalized" and "rolling", the model outputs a scalar regression value.
        dropout_prob: Dropout probability applied after fc1 and fc2. Dropout remains active (MC Dropout) even during evaluation for uncertainty estimation.
        logit_clamp_min, logit_clamp_max: Clamp values for logits in binary mode during evaluation.
        """
        super(FusionNet, self).__init__()
        self.use_attention = use_attention
        self.fusion_method = fusion_method
        self.target_mode = target_mode
        self.dropout_prob = dropout_prob
        self.logit_clamp_min = logit_clamp_min
        self.logit_clamp_max = logit_clamp_max
        
        if fusion_method == 'concat':
            fc1_in_dim = input_dim + 768
        elif fusion_method == 'average':
            fc1_in_dim = input_dim
        else:
            raise ValueError("Invalid fusion_method; choose 'concat' or 'average'.")
        
        self.fc1 = nn.Linear(fc1_in_dim, hidden_dim)
        if self.use_attention:
            self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=1, batch_first=True)
        else:
            self.attention = None
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        
        self.apply(init_weights)
    
    def forward(self, emb1, emb2):
        # Fusion: average or concatenate.
        if self.fusion_method == 'average':
            x = (emb1 + emb2) / 2.0
        else:
            x = torch.cat((emb1, emb2), dim=1)
        
        if torch.isnan(x).any():
            print("NaN detected after fusion, x shape:", x.shape)
        
        # First FC layer + ReLU.
        h = self.fc1(x)
        h = self.relu(h)
        # MC Dropout: always active.
        h = F.dropout(h, p=self.dropout_prob, training=True)
        
        # Optional attention mechanism.
        if self.use_attention:
            h_seq = h.unsqueeze(1)
            if torch.isnan(h_seq).any():
                print("NaN detected before attention (h_seq)")
            attn_output, _ = self.attention(h_seq, h_seq, h_seq)
            h = attn_output.squeeze(1)
            if torch.isnan(h).any():
                print("NaN detected after attention squeeze")
        
        # Second FC layer + ReLU.
        h2 = self.fc2(h)
        h2 = self.relu(h2)
        h2 = F.dropout(h2, p=self.dropout_prob, training=True)
        if torch.isnan(h2).any():
            print("NaN detected after fc2 and dropout")
        
        # Final output layer.
        output = self.out(h2)
        if torch.isnan(output).any():
            print("NaN detected after output layer")
        
        # For binary mode during evaluation, clamp logits and apply sigmoid.
        if self.target_mode == "binary" and not self.training:
            output = torch.clamp(output, min=self.logit_clamp_min, max=self.logit_clamp_max)
            output = torch.sigmoid(output)
        
        return output.view(-1)

    def predict(self, x1, x2):
        self.eval()
        if not isinstance(x1, torch.Tensor):
            x1 = torch.from_numpy(x1.astype(np.float32))
        if not isinstance(x2, torch.Tensor):
            x2 = torch.from_numpy(x2.astype(np.float32))
        with torch.no_grad():
            y = self.forward(x1.unsqueeze(0), x2.unsqueeze(0))
        return y.squeeze(0).cpu().numpy()

    def save_model(self, filepath):
        torch.save(self.state_dict(), filepath)
    
    def load_model(self, filepath):
        self.load_state_dict(torch.load(filepath))
        self.eval()

if __name__ == "__main__":
    # Quick test.
    dummy_emb1 = torch.randn(32, 10)
    dummy_emb2 = torch.randn(32, 768)
    model = FusionNet(input_dim=10, hidden_dim=512, use_attention=True, fusion_method='concat', target_mode="binary")
    output = model(dummy_emb1, dummy_emb2)
    print("Test output:", output)