
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv
import matplotlib.pyplot as plt

# # === Step 1: Load Dataset ===
# datafile = "topopt_dataset.npz"
# data = np.load(datafile)
# density = data['density']      # shape: (T, 30, 50)
# energy = data['energy']        # shape: (T, 30, 50)
# next_density = data['next_density']  # shape: (T, 30, 50)


# === Step 1: Load and Combine Datasets ===
data_std = np.load("topopt_dataset.npz")
data_llm = np.load("topopt_dataset_llm.npz")

# Concatenate data along the first axis (time/samples)
density = np.concatenate([data_std['density'], data_llm['density']], axis=0)
energy = np.concatenate([data_std['energy'], data_llm['energy']], axis=0)
next_density = np.concatenate([data_std['next_density'], data_llm['next_density']], axis=0)


num_samples, H, W = density.shape
N = H * W  # number of nodes

# === Step 2: Build Grid Edge Index ===
def build_grid_edges(h, w):
    edges = []
    for i in range(h):
        for j in range(w):
            idx = i * w + j
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # 4-neighbors
                ni, nj = i + di, j + dj
                if 0 <= ni < h and 0 <= nj < w:
                    nidx = ni * w + nj
                    edges.append((idx, nidx))
    return torch.tensor(edges, dtype=torch.long).t()  # shape (2, num_edges)

edge_index = build_grid_edges(H, W)

# === Step 3: Convert Dataset to PyG Format ===
graph_data_list = []

for t in range(num_samples - 1):
    x_density = density[t].flatten()
    x_energy = energy[t].flatten()
    x = np.stack([x_density, x_energy], axis=1)  # (N, 2)
    y = next_density[t + 1].flatten()  # target is next step density

    graph = Data(
        x=torch.tensor(x, dtype=torch.float),
        edge_index=edge_index,
        y=torch.tensor(y, dtype=torch.float)
    )
    graph_data_list.append(graph)

# === Step 4: Data Loader ===
loader = DataLoader(graph_data_list, batch_size=4, shuffle=True)

# === Step 5: Define GNN Model ===
class TopOptGNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(2, 64)
        self.conv2 = GCNConv(64, 64)
        self.fc = torch.nn.Linear(64, 1)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return torch.sigmoid(self.fc(x)).squeeze()

# === Step 6: Train the GNN ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TopOptGNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(1, 201):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index)
        loss = F.mse_loss(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch:03d}, Loss: {total_loss:.4f}")

# === Step 7: Visualize a Prediction ===
model.eval()
sample = graph_data_list[0].to(device)
with torch.no_grad():
    pred = model(sample.x, sample.edge_index).cpu().numpy()

torch.save(model.state_dict(), "topopt_gnn_model_new.pth")
print("✅ Model saved to 'topopt_gnn_model_new.pth'")

# Plot input θ_t, target θ_{t+1}, prediction
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(sample.x[:, 0].cpu().numpy().reshape(H, W), cmap='bone_r')
plt.title("Input θ_t")

plt.subplot(1, 3, 2)
plt.imshow(sample.y.cpu().numpy().reshape(H, W), cmap='bone_r')
plt.title("True θ_{t+1}")

plt.subplot(1, 3, 3)
plt.imshow(pred.reshape(H, W), cmap='bone_r')
plt.title("Predicted θ_{t+1}")

plt.tight_layout()
plt.show()

