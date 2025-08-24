# Fluid Flow GNN

Graph neural network that learns to predict fluid dynamics around cylinders from computational mesh data.

## What it does

Takes a mesh snapshot of fluid flow and predicts what the flow will look like several timesteps later. The mesh gets converted into a graph where each mesh point becomes a node and connections between points become edges.

## The Model

Three main parts working together:

**GNNEncoder** - Takes raw mesh data and converts it to useful representations  
**GNNProcessor** - Does the heavy lifting with message passing between neighboring mesh points  
**GNNDecoder** - Converts the processed information back to flow predictions  

The processor runs multiple rounds of message passing where each mesh point talks to its neighbors and updates its understanding of the local flow.

## How the data works

Starting with mesh simulation data, we build graphs like this:
- Each mesh point becomes a node with flow properties (velocity, pressure, etc.)
- Mesh connectivity becomes edges between nodes  
- We add extra info like mass and simulation parameters
- Target is the flow state 5 timesteps in the future

## Quick start

```python
# Load your data
train_graphs = torch.load("processed/train_graphs_stride5.pt") 
test_graphs = torch.load("processed/test_graphs_stride5.pt")

# Build the model
model = GNNModel(
    in_channels=feature_size,
    hidden_dim=128, 
    out_channels=flow_vars,
    num_layers=3
)

# Train it
for batch in dataloader:
    prediction, _ = model(batch)
    loss = F.mse_loss(prediction, batch.y)
```

## What you need

Your data should have:
```
x - flow sequences [rollouts, time, nodes, features]  
edge_index - which nodes connect to which [2, edges]
edge_attr - edge properties [edges, features]
mass - node masses [nodes]  
para - simulation parameters [rollouts]
```

## Dependencies
- torch + torch-geometric for the GNN stuff
- numpy for data handling  
- tqdm for progress bars
- sklearn for train/test splits

## Why this approach

Traditional CFD is slow. This learns the physics from data and can predict flow evolution much faster than solving the full equations. Good for getting quick approximations or initializing more detailed simulations.
