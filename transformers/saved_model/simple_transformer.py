import torch.nn as nn

class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size=50, embed_dim=32, num_heads=2, num_layers=1, num_outputs=6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.ModuleList([nn.Linear(embed_dim, vocab_size) for _ in range(num_outputs)])
        self.num_outputs = num_outputs
    def forward(self, x):
        batch_size, seq_len, num_features = x.shape
        x = self.embedding(x)  # [batch_size, seq_len, 7, embed_dim]
        x = x.mean(dim=2)      # [batch_size, seq_len, embed_dim]
        x = self.transformer(x)
        x = x[:, -1, :]        # use last token's output
        outs = [fc(x) for fc in self.fc]  # list of [batch_size, vocab_size]
        return outs
