import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm
import os
from sklearn.model_selection import train_test_split

data = np.load("dataset/rawData.npy", allow_pickle=True)
x_all = data["x"]
edge_index = torch.tensor(data["edge_index"], dtype=torch.long)
edge_attr = torch.tensor(data["edge_attr"], dtype=torch.float)
mass = torch.tensor(data["mass"], dtype=torch.float)
para = data["para"]

stride = 5
graph_data = []

for rollout in tqdm(range(len(x_all)), desc="Building rollout segments"):
    para_tensor = torch.tensor([para[rollout]], dtype=torch.float)

    for t in range(0, x_all.shape[1] - stride, stride):
        x_t = torch.tensor(x_all[rollout, t], dtype=torch.float)
        x_tH = torch.tensor(x_all[rollout, t + stride], dtype=torch.float)

        param_repeat = para_tensor.repeat(x_t.shape[0], 1)
        x_input = torch.cat([x_t, mass, param_repeat], dim=1)

        data_obj = Data(
            x=x_input,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=x_tH,
            para=para_tensor
        )
        graph_data.append(data_obj)


train_graphs, test_graphs = train_test_split(graph_data, test_size=0.2, random_state=42)
os.makedirs("processed", exist_ok=True)
torch.save(train_graphs, "processed/train_graphs_stride5.pt")
torch.save(test_graphs, "processed/test_graphs_stride5.pt")

