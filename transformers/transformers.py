"""
I have a dataset that consists of few thousands observations in and Excel .csv file. 
The Excel csv file has columns headings in the first row. 
Each subsequent row represents a single observations and it has 
an observation date in the column named 'date'. 
The subsequent columns are named 'd1', 'd2','d3','d4','d5','d6','bonus' 
and contain categorical values. 
There are 50 categories represented by numbers from 0 till 49. 
A single row may not have repeating categories.
Create python code to input the data file, prepare it for processing, 
then create a transformer model to train on the created sequences, 
and then predict next observation values in columns d1 - bonus. 
Let me know if you need additional instructions or information. 
"""
import os
from datetime import datetime
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# Load and preprocess the dataset
# df = pd.read_csv("../data/2025.csv")
df = pd.read_csv("data/data_all_l649.csv")
cols = ['d1', 'd2', 'd3', 'd4', 'd5', 'd6', 'bonus']
df[cols] = df[cols].map(lambda x: int(str(x).strip()))
df = df[df[cols].apply(lambda row: len(set(row)) == len(row), axis=1)].reset_index(drop=True)

# Create sequences of 10 observations to predict the next one
sequence_length = 10 # pylint: disable=invalid-name
sequences = []
targets = []

for i in range(len(df) - sequence_length):
    seq = df.iloc[i:i+sequence_length][cols].values.flatten()
    tgt = df.iloc[i+sequence_length][cols].values
    sequences.append(seq)
    targets.append(tgt)

# Convert to tensors
X = torch.tensor(sequences, dtype=torch.long)
y = torch.tensor(targets, dtype=torch.long)

# Custom Dataset
class LotteryDataset(Dataset):
    """Custom Dataset for Lottery sequences."""
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

dataset = LotteryDataset(X, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Transformer Model
class LotteryTransformer(nn.Module):
    """Transformer model for Lottery data."""
    def __init__(self, vocab_size=50, embed_dim=64, num_heads=4, num_layers=2, seq_len=70, num_outputs=7):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.ModuleList([nn.Linear(embed_dim, vocab_size) for _ in range(num_outputs)])
        self.seq_len = seq_len
    def forward(self, x): # pylint: disable=invalid-name
        x = self.embedding(x)
        x = self.transformer(x)
        x = x[:, -1, :]  # use last token's output
        return [fc(x) for fc in self.fc]

# Initialize model, loss, optimizer
model = LotteryTransformer()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Device handling: use CUDA if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# Move model to device
model = model.to(device)

# Training loop
losses = []
for epoch in range(25):
    total_loss = 0  # pylint: disable=invalid-name
    for batch_x, batch_y in dataloader:
        # Move batch tensors to device
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = sum([criterion(output, batch_y[:, i]) for i, output in enumerate(outputs)])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    losses.append(avg_loss)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

# Plot training loss
plt.figure(figsize=(10, 5))
plt.plot(range(1, 26), losses, marker='o')
plt.title("Training Loss Over 25 Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
# Ensure results directory exists (resolve relative to repository root) and normalize path
results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
os.makedirs(results_dir, exist_ok=True)
save_path = os.path.join(results_dir, "training_loss.png")
try:
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved training loss plot to: {save_path}")
except Exception as e:
    print(f"Error saving training loss plot to {save_path}: {e}")
plt.close()

# Inference on last 10 sequences
last_seq = df.iloc[-sequence_length:][cols].values.flatten()
input_tensor = torch.tensor(last_seq, dtype=torch.long).unsqueeze(0)
with torch.no_grad():
    # Move input to device for inference
    input_tensor = input_tensor.to(device)
    predictions = model(input_tensor)
    predicted_values = [torch.argmax(p, dim=1).cpu().item() for p in predictions]

# Save predictions
preds_path = os.path.join(results_dir, "predicted_values.txt")
with open(preds_path, "w", encoding="utf-8") as f:
    f.write(f"Current date: {datetime.now()} \n")
    f.write("Predicted next observation:\n")
    for col, val in zip(cols, predicted_values):
        f.write(f"{col}: {val}\n")
if os.path.exists(preds_path):
    print(f"Saved predictions to: {preds_path}")
else:
    print(f"Failed to save predictions to: {preds_path}")
