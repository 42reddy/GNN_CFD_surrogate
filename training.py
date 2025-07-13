import torch
from GNN import GNNModel
from torch_geometric.data import Data, DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

train_graphs = torch.load("processed/train_graphs_stride5.pt", weights_only=False)
test_graphs = torch.load("processed/train_graphs_stride5.pt",weights_only=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GNNModel(in_channels=5, hidden_dim=4, out_channels=3, num_layers=1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
criterion = torch.nn.MSELoss()

train_loader = DataLoader(train_graphs, batch_size=128, shuffle=True)
test_loader = DataLoader(test_graphs, batch_size=128, shuffle=False)

num_epochs = 10

for epoch in range(num_epochs):

    model.train()
    total_loss = 0.0

    train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1:02d} [Train]", leave=False)
    for batch in train_bar:
        batch = batch.to(device)
        optimizer.zero_grad()

        x = batch.x.clone()
        edge_index, edge_attr = batch.edge_index, batch.edge_attr
        target = batch.y  # Final field at t = 400

        for _ in range(0, 400, 5):
            out = model(Data(x=x, edge_index=edge_index, edge_attr=edge_attr))
            x = torch.cat([out.detach(), x[:, 3:]], dim=1)  # detach and keep [mass, para]

        # Compare final predicted x to ground truth
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        train_bar.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(train_loader)

    # Evaluation
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        test_bar = tqdm(test_loader, desc=f"Epoch {epoch+1:02d} [Test]", leave=False)
        for batch in test_bar:
            batch = batch.to(device)

            x = batch.x.clone()
            edge_index, edge_attr = batch.edge_index, batch.edge_attr
            target = batch.y  # Final target

            for _ in range(0, 400, 15):
                out = model(Data(x=x, edge_index=edge_index, edge_attr=edge_attr))
                x = torch.cat([out.detach(), x[:, 3:]], dim=1)

            loss = criterion(out, target)
            test_loss += loss.item()
            test_bar.set_postfix(loss=loss.item())

    test_loss /= len(test_loader)
    print(f"Epoch {epoch+1:02d} | Train Loss: {avg_loss:.6f} | Test Loss: {test_loss:.6f}")









mesh_pos = np.loadtxt("dataset/meshPosition_all.txt")

model.eval()
sample_indices = [0, 1, 2]
num_samples = len(sample_indices)

fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples), constrained_layout=True)

if num_samples == 1:
    axes = np.expand_dims(axes, axis=0)

with torch.no_grad():
    for i, idx in enumerate(sample_indices):
        graph = test_graphs[idx].to(device)
        pred = model(graph)

        y_pred = pred
        y_true = graph.y

        v_pred = y_pred[:, :2]
        v_true = y_true[:, :2]

        mag_pred = torch.norm(v_pred, dim=1).cpu().numpy()
        mag_true = torch.norm(v_true, dim=1).cpu().numpy()
        abs_err = np.abs(mag_pred - mag_true)

        # === Plotting ===
        # 1. Predicted
        cf1 = axes[i, 0].tricontourf(mesh_pos[:, 0], mesh_pos[:, 1], mag_pred, levels=100, cmap='viridis')
        axes[i, 0].set_title(f"Sample {idx} - Predicted")
        axes[i, 0].axis("equal")
        fig.colorbar(cf1, ax=axes[i, 0])

        # 2. Ground Truth
        cf2 = axes[i, 1].tricontourf(mesh_pos[:, 0], mesh_pos[:, 1], mag_true, levels=100, cmap='viridis')
        axes[i, 1].set_title(f"Sample {idx} - Ground Truth")
        axes[i, 1].axis("equal")
        fig.colorbar(cf2, ax=axes[i, 1])

        # 3. Absolute Error
        cf3 = axes[i, 2].tricontourf(mesh_pos[:, 0], mesh_pos[:, 1], abs_err, levels=100, cmap='inferno')
        axes[i, 2].set_title(f"Sample {idx} - Abs Error")
        axes[i, 2].axis("equal")
        fig.colorbar(cf3, ax=axes[i, 2])

plt.show()

