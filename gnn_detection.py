import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import numpy as np

# Example detected objects: [x_min, y_min, x_max, y_max, class_id]
detected_objects = np.array([
    [10, 20, 50, 70, 0],  # Object 1 (e.g., a car)
    [30, 40, 60, 80, 1],  # Object 2 (e.g., a dog)
    [55, 60, 90, 100, 0], # Object 3 (e.g., a cat)
])

# Function to create edges based on spatial proximity
def create_edges(objects):
    edges = []
    for i in range(len(objects)):
        for j in range(len(objects)):
            if i != j:
                # Check if objects are close to each other
                if (objects[i][0] < objects[j][2] and objects[i][2] > objects[j][0] and
                    objects[i][1] < objects[j][3] and objects[i][3] > objects[j][1]):
                    edges.append((i, j))
    return torch.tensor(edges, dtype=torch.long).t().contiguous()

# Create node features (bounding box coordinates + class)
features = torch.tensor(detected_objects[:, :4], dtype=torch.float)  # Using only the bounding boxes for simplicity
edge_index = create_edges(detected_objects)

# Create a Data object for PyTorch Geometric
data = Data(x=features, edge_index=edge_index)

# Define the GNN model
class GNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x  # Output the refined features

# Model parameters
input_dim = 4  # Number of features (bounding box coordinates)
hidden_dim = 8
output_dim = 2  # For example, refining the class probabilities

# Initialize the GNN
model = GNN(input_dim, hidden_dim, output_dim)
output = model(data)

# Print the refined features for each detected object
print("Refined features for detected objects:")
print(output.detach().numpy())
