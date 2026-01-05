import torch.nn as nn

class SimpleLSTM(nn.Module):
    def __init__(self, vocab_size=50, embed_dim=64, hidden_dim=128, num_layers=1, num_outputs=6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.ModuleList([nn.Linear(hidden_dim, vocab_size) for _ in range(num_outputs)])
        self.num_outputs = num_outputs
    def forward(self, x):
        batch_size, seq_len, num_features = x.shape
        x = self.embedding(x)  # [batch_size, seq_len, 7, embed_dim]
        x = x.mean(dim=2)      # [batch_size, seq_len, embed_dim]
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]  # use last time step's output
        outs = [fc(x) for fc in self.fc]  # list of [batch_size, vocab_size]
        return outs
