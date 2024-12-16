from torch.optim import Optimizer
import pickle
import torch
import torch.nn as nn
import neat
import numpy as np
from sklearn.metrics import roc_auc_score
from Preprocessor import DataPreprocessor

from torch.optim import Optimizer
import pickle
import torch
import torch.nn as nn
import neat
import numpy as np
from sklearn.metrics import roc_auc_score

from torch.optim import Optimizer
import pickle
import torch
import torch.nn as nn
import neat
import numpy as np
from sklearn.metrics import roc_auc_score

from torch.optim import Optimizer
import pickle
import torch
import torch.nn as nn
import neat
import numpy as np
from collections import defaultdict, deque

class NEATToTorch(nn.Module):
    def __init__(self, genome, config):
        super(NEATToTorch, self).__init__()
        
        # Create NEAT network from genome
        self.neat_network = neat.nn.FeedForwardNetwork.create(genome, config)
        
        # Store network information
        self.input_nodes = set(config.genome_config.input_keys)
        self.output_nodes = set(config.genome_config.output_keys)
        self.node_evals = self.neat_network.node_evals
        
        print(f"Input nodes: {self.input_nodes}")
        print(f"Output nodes: {self.output_nodes}")
        print("\nNode evaluations:")
        for eval in self.node_evals:
            print(f"Node {eval[0]}: {eval}")
        
        # Create dictionaries to store node information
        self.node_layers = {}
        self.layer_nodes = {}
        self.active_nodes = set()
        
        # Initialize layers
        self.layers = nn.ModuleDict()
        self.activation_functions = nn.ModuleDict()
        
        # Build network structure
        self.find_active_nodes()
        print(f"\nActive nodes: {self.active_nodes}")
        print(f"Node layers: {self.node_layers}")
        print(f"Layer nodes: {self.layer_nodes}")
        
        self.build_pytorch_layers()
        self.transfer_weights()
        
        # Print parameter information
        print("\nModel parameters:")
        total_params = 0
        for name, param in self.named_parameters():
            print(f"{name}: shape {param.shape}, requires_grad={param.requires_grad}")
            total_params += param.numel()
        print(f"Total trainable parameters: {total_params}")


    def find_active_nodes(self):
        """Find all nodes that are part of valid paths from inputs to outputs"""
        # Build forward and backward connection maps
        forward_connections = defaultdict(set)
        backward_connections = defaultdict(set)
        node_info = {}  # Store node evaluation information
        
        # First pass: build connection maps
        for node_eval in self.node_evals:
            node_id = node_eval[0]
            node_info[node_id] = node_eval
            
            if len(node_eval) > 3:
                inputs = node_eval[3]
                if isinstance(inputs, list):
                    for input_id, _ in inputs:
                        forward_connections[input_id].add(node_id)
                        backward_connections[node_id].add(input_id)
                elif isinstance(inputs, dict):
                    for input_id in inputs:
                        forward_connections[input_id].add(node_id)
                        backward_connections[node_id].add(input_id)

        # Find nodes reachable from inputs
        reachable_from_inputs = set()
        queue = deque(self.input_nodes)
        while queue:
            node = queue.popleft()
            reachable_from_inputs.add(node)
            for next_node in forward_connections[node]:
                if next_node not in reachable_from_inputs:
                    queue.append(next_node)

        # Find nodes that can reach outputs
        can_reach_outputs = set()
        queue = deque(self.output_nodes)
        while queue:
            node = queue.popleft()
            can_reach_outputs.add(node)
            for prev_node in backward_connections[node]:
                if prev_node not in can_reach_outputs:
                    queue.append(prev_node)

        # Active nodes are those that are both reachable from inputs and can reach outputs
        self.active_nodes = reachable_from_inputs & can_reach_outputs

        # Now assign layers to active nodes
        active_inputs = self.input_nodes & self.active_nodes
        self.node_layers = {node: 0 for node in active_inputs}
        self.layer_nodes = {0: list(active_inputs)}
        
        # Topologically sort remaining active nodes
        remaining_nodes = self.active_nodes - active_inputs
        layer = 1
        
        while remaining_nodes:
            nodes_in_layer = set()
            for node in remaining_nodes:
                if node in backward_connections:
                    input_nodes = backward_connections[node]
                    if all(input_node in self.node_layers for input_node in input_nodes):
                        nodes_in_layer.add(node)
                        self.node_layers[node] = layer
            
            if not nodes_in_layer:
                break
                
            self.layer_nodes[layer] = list(nodes_in_layer)
            remaining_nodes -= nodes_in_layer
            layer += 1

    def build_pytorch_layers(self):
        """Convert NEAT network structure to PyTorch layers"""
        print("\nBuilding PyTorch layers:")
        if not self.layer_nodes:
            print("No layer nodes found!")
            return
        
        max_layer = max(self.layer_nodes.keys())
        for layer in range(max_layer):
            if layer in self.layer_nodes and layer + 1 in self.layer_nodes:
                input_size = len(self.layer_nodes[layer])
                output_size = len(self.layer_nodes[layer + 1])
                
                print(f"\nLayer {layer}:")
                print(f"Input size: {input_size}")
                print(f"Output size: {output_size}")
                print(f"Input nodes: {self.layer_nodes[layer]}")
                print(f"Output nodes: {self.layer_nodes[layer + 1]}")
                
                if input_size > 0 and output_size > 0:
                    self.layers[f'linear_{layer}'] = nn.Linear(input_size, output_size)
                    self.activation_functions[f'activation_{layer}'] = nn.Sigmoid()
                    
                    # Get activation function for next layer's nodes
                    node_id = self.layer_nodes[layer + 1][0]
                    for node_eval in self.node_evals:
                        if node_eval[0] == node_id:
                            if node_eval[2] == neat.activations.sigmoid_activation:
                                self.activation_functions[f'activation_{layer}'] = nn.Sigmoid()
                            else:
                                self.activation_functions[f'activation_{layer}'] = nn.ReLU()
                            break

    def forward(self, x):
        """Forward pass through the network"""
        current_output = x
        
        for layer_idx in range(len(self.layer_nodes) - 1):
            if f'linear_{layer_idx}' in self.layers:
                current_output = self.layers[f'linear_{layer_idx}'](current_output)
                current_output = self.activation_functions[f'activation_{layer_idx}'](current_output)
        
        return torch.sigmoid(current_output)
    
    def transfer_weights(self):
        """Transfer weights from NEAT network to PyTorch layers"""
        print("\nTransferring weights:")
        if not self.layer_nodes:
            print("No layers to transfer weights to!")
            return
            
        node_connections = {}
        node_biases = {}
        
        # Extract connection weights and biases
        for node_eval in self.node_evals:
            node = node_eval[0]
            if node not in self.active_nodes:
                continue
                
            if len(node_eval) > 3:
                inputs = node_eval[3]
                if isinstance(inputs, list):
                    node_connections[node] = {inp[0]: inp[1] for inp in inputs 
                                           if inp[0] in self.active_nodes}
                elif isinstance(inputs, dict):
                    node_connections[node] = {k: v for k, v in inputs.items() 
                                           if k in self.active_nodes}
            
            if len(node_eval) > 4:
                node_biases[node] = node_eval[4]
        
        print("\nNode connections:")
        for node, connections in node_connections.items():
            print(f"Node {node}: {connections}")
        
        print("\nNode biases:")
        print(node_biases)
        
        # Transfer weights layer by layer
        for layer_idx in range(len(self.layer_nodes) - 1):
            if f'linear_{layer_idx}' in self.layers:
                layer = self.layers[f'linear_{layer_idx}']
                prev_layer_nodes = self.layer_nodes[layer_idx]
                current_layer_nodes = self.layer_nodes[layer_idx + 1]
                
                print(f"\nTransferring weights for layer {layer_idx}:")
                print(f"Previous layer nodes: {prev_layer_nodes}")
                print(f"Current layer nodes: {current_layer_nodes}")
                
                weights = torch.zeros((len(current_layer_nodes), len(prev_layer_nodes)))
                biases = torch.zeros(len(current_layer_nodes))
                
                for i, target_node in enumerate(current_layer_nodes):
                    if target_node in node_connections:
                        for j, source_node in enumerate(prev_layer_nodes):
                            if source_node in node_connections[target_node]:
                                weight = node_connections[target_node][source_node]
                                weights[i, j] = weight
                                print(f"Setting weight from {source_node} to {target_node}: {weight}")
                    
                    if target_node in node_biases:
                        bias = node_biases[target_node]
                        biases[i] = bias
                        print(f"Setting bias for node {target_node}: {bias}")
                
                with torch.no_grad():
                    layer.weight.copy_(weights)
                    layer.bias.copy_(biases)
                    print(f"Layer {layer_idx} weights shape: {layer.weight.shape}")
                    print(f"Layer {layer_idx} weights:\n{layer.weight}")
                    print(f"Layer {layer_idx} biases: {layer.bias}")

def load_neat_as_pytorch(genome, config):
    """Convert NEAT genome to PyTorch model"""
    model = NEATToTorch(genome, config)
    model.eval()  # Set to evaluation mode
    return model

# Example usage
def main():
    # Load NEAT configuration
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-neat')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        config_path)
    
    # Load your trained NEAT genome
    with open('winner-neat.pkl', 'rb') as f:
        genome = pickle.load(f)
    
    # Convert to PyTorch model
    pytorch_model = load_neat_as_pytorch(genome, config)
    
    return pytorch_model


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import neat
import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split

# Custom Dataset class that uses the preprocessor
class PreprocessedDataset(Dataset):
    def __init__(self, X, y, preprocessor=None):
        self.raw_X = X
        if preprocessor:
            # Transform each row using the preprocessor
            processed_data = [preprocessor.transform_row(row) for _, row in X.iterrows()]
            self.X = torch.FloatTensor(processed_data)
        else:
            # Convert X to float tensor, handling different types
            if isinstance(X, pd.DataFrame):
                X = X.values
            self.X = torch.FloatTensor(X.astype(float))

        # Convert y to float tensor, handling different types
        if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
            y = y.values
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        self.y = torch.FloatTensor(y.astype(float))
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


    # Training loop with validation
def train_with_validation(model, train_loader, val_loader, epochs, learning_rate=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_x, batch_y in train_loader:
            # Convert string labels to numeric if needed
            if isinstance(batch_y[0].item(), str):
                batch_y = torch.tensor([(1 if label == 'Y' else 0) for label in batch_y]).float()
            
            optimizer.zero_grad()
            output = model(batch_x)
            output = output.view(-1)  # Flatten output
            loss = criterion(output, batch_y.float())
            loss.backward()
            optimizer.step()
            
            # Simple thresholding for predictions
            predictions = (output > 0.5).float()
            total_train_loss += loss.item()
            train_correct += (predictions == batch_y).sum().item()
            train_total += batch_y.size(0)
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                # Convert string labels to numeric if needed
                if isinstance(batch_y[0].item(), str):
                    batch_y = torch.tensor([(1 if label == 'Y' else 0) for label in batch_y]).float()
                
                output = model(batch_x)
                output = output.view(-1)
                # Simple thresholding without sigmoid
                predictions = (output > 0.5).float()
                val_loss = criterion(output, batch_y.float())
                
                total_val_loss += val_loss.item()
                val_correct += (predictions == batch_y).sum().item()
                val_total += batch_y.size(0)
        
        # Calculate metrics
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        train_accuracy = 100 * train_correct / train_total
        val_accuracy = 100 * val_correct / val_total
        
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'  Training Loss: {avg_train_loss:.4f}')
        print(f'  Validation Loss: {avg_val_loss:.4f}')



def prepare_data_and_train(data_df, target_column, neat_model_path, batch_size=32, epochs=100):
    """
    Complete pipeline for preprocessing data and training the converted NEAT model
    
    Args:
        data_df: pandas DataFrame containing your data
        target_column: name of the target column
        neat_model_path: path to your saved NEAT model pickle file
        batch_size: batch size for training
        epochs: number of training epochs
        
    Returns:
        tuple: (trained PyTorch model, preprocessor)
    """
    # Split into features and target
    X = data_df.drop(columns=[target_column])
    y = data_df[target_column]
    
    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and fit the preprocessor on training data
    preprocessor = DataPreprocessor(X_train)
    
    # Create datasets
    train_dataset = PreprocessedDataset(X_train, y_train.to_frame(), preprocessor)
    val_dataset = PreprocessedDataset(X_val, y_val.to_frame(), preprocessor)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Load and convert NEAT model
    with open(neat_model_path, 'rb') as f:
        neat_winner = pickle.load(f)
    
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                    neat.DefaultSpeciesSet, neat.DefaultStagnation,
                    r'c:\Users\engin\OneDrive\Desktop\GITHUB\config-feedforward')
    
    pytorch_model = NEATToTorch(neat_winner, config)
    
    # Verify model has parameters before training
    trainable_params = list(pytorch_model.parameters())
    if not trainable_params:
        raise ValueError("Model has no trainable parameters!")
    
    print(f"Number of trainable parameters: {sum(p.numel() for p in trainable_params)}")
    
    # Train the model
    train_with_validation(pytorch_model, train_loader, val_loader, epochs)
    
    return pytorch_model, preprocessor

def main():
    # Load your data
    data_df = pd.read_csv(r'c:\Users\engin\OneDrive\Desktop\DATA201\Project\train-1.csv')
    data_df['Loan_Status'] = data_df['Loan_Status'].map({'Y': 1, 'N': 0})
    data_df.drop('Loan_ID', axis=1, inplace=True)
    
    # Train the model and get both model and preprocessor
    model, preprocessor = prepare_data_and_train(
        data_df=data_df,
        target_column='Loan_Status',
        neat_model_path=r'c:\Users\engin\OneDrive\Desktop\GITHUB\neat_outputs\winner.pkl',
        batch_size=32,
        epochs=50
    )
    
    # Load and preprocess test data
    new_data = pd.read_csv(r'c:\Users\engin\OneDrive\Desktop\DATA201\Project\test-1.csv')
    new_data.dropna(inplace=True)
    
    # Make predictions
    y_true = []
    y_scores = []
    for idx in range(len(new_data)):
        row = new_data.iloc[idx]
        transformed_row = preprocessor.transform_row(row)
        output = model(torch.FloatTensor(transformed_row))
        y_scores.append(output.item())
        y_true.append(row['Loan_Status'])

    # Calculate metrics
    y_true = [1 if y == 'Y' else 0 for y in y_true]
    auc_roc = roc_auc_score(y_true, y_scores)
    correct_predictions = sum(1 for true, score in zip(y_true, y_scores) 
                            if (true == 1 and score > 0.5) or (true == 0 and score <= 0.5))
    accuracy = correct_predictions / len(y_true)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC-ROC: {auc_roc:.4f}")
    from torchviz import make_dot
    import hiddenlayer as hl
    inputs = torch.FloatTensor(transformed_row)
    transforms = [ hl.transforms.Prune('Constant') ] 
    graph = hl.build_graph(model, inputs, transforms=transforms)
    graph.theme = hl.graph.THEMES['blue'].copy()
    graph.save('modelback', format='png')
    make_dot(output.mean(), params=dict(model.named_parameters())).render("attached", format="png")

if __name__ == '__main__':
    main()