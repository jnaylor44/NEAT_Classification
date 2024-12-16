from Environment import Environment
from Preprocessor import DataPreprocessor, OutlierRemover
from Organism import Organism
from DataRow import DataRow
import random
import numpy as np
import neat
import os
import copy
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
import pickle
import visualize

class Substrate:
    def __init__(self, input_dimensions, hidden_dimensions, output_dimensions):
        """
        Initialize the substrate (the geometric layout of nodes)
        
        Args:
            input_dimensions: Tuple of (width, height) for input layer
            hidden_dimensions: List of tuples for hidden layers
            output_dimensions: Tuple of (width, height) for output layer
        """
        self.input_dims = input_dimensions
        self.hidden_dims = hidden_dimensions
        self.output_dims = output_dimensions
        
        # Generate node coordinates
        self.input_coords = self._generate_coordinates(input_dimensions, -1.0)
        
        # Calculate z-coordinates for hidden layers
        num_hidden = len(hidden_dimensions)
        if num_hidden > 0:
            z_coords = np.linspace(-0.5, 0.5, num_hidden)
            self.hidden_coords = [self._generate_coordinates(dims, z) 
                                for dims, z in zip(hidden_dimensions, z_coords)]
        else:
            self.hidden_coords = []
            
        self.output_coords = self._generate_coordinates(output_dimensions, 1.0)

    def _generate_coordinates(self, dimensions, z_coord):
        """Generate coordinates for nodes in a layer"""
        coords = []
        width, height = dimensions
        for x in np.linspace(-1.0, 1.0, width):
            for y in np.linspace(-1.0, 1.0, height):
                coords.append((x, y, z_coord))
        return coords


class HyperNEAT:
    def __init__(self, config_path, dataset_path):
        """
        Initialize HyperNEAT with configuration and dataset paths
        """
        self.config_path = config_path
        self.dataset_path = dataset_path
        self.config = None
        self.env = None
        self.stats = None
        self.output_dir = "hyper_neat_outputs"
        os.makedirs(self.output_dir, exist_ok=True)

        temp_env = Environment(None, dataset_path)
        num_features = len(temp_env.header)
        
        # width = int(np.ceil(np.sqrt(num_features)))
        # height = int(np.ceil(num_features / width))
        input_dims = (11, 1)
        hidden_dims = [(4, 4), (3, 3), (2, 2)]
        output_dims = (1, 1)
        
        self.substrate = Substrate(input_dims, hidden_dims, output_dims)
        self.weight_threshold = 0.1

    def run(self, generations=15):
        """
        Run HyperNEAT evolution
        """
        self.setup_config()
        
        # Initialize environment
        self.env = Environment(self.config, self.dataset_path)
        
        p = neat.Population(self.config)
        p.add_reporter(neat.StdOutReporter(True))
        self.stats = neat.StatisticsReporter()
        p.add_reporter(self.stats)
        p.add_reporter(neat.Checkpointer(50))
        
        winner = p.run(self.eval_genomes, generations)
        
        # Save results
        winner_path = os.path.join(self.output_dir, "winner_cppn.pkl")
        preprocessor_path = os.path.join(self.output_dir, "preprocessor.pkl")
        
        with open(winner_path, "wb") as f:
            pickle.dump(winner, f)
        with open(preprocessor_path, "wb") as f:
            pickle.dump(self.env.preprocessor, f)
        
        self._visualize_results(winner)
        
        return winner
    
    def setup_config(self):
        """
        Set up NEAT configuration for CPPN
        """
        self.config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            self.config_path
        )

    def query_cppn(self, cppn, source_coords, target_coords):
        """
        Query CPPN for connection weight between two points
        
        The CPPN input now includes:
        - x1, y1, z1 (source coordinates)
        - x2, y2, z2 (target coordinates)
        - delta_x, delta_y, delta_z (coordinate differences)
        - distance (Euclidean distance between points)
        - bias (1.0)
        """
        x1, y1, z1 = source_coords
        x2, y2, z2 = target_coords
        
        # Calculate additional inputs
        delta_x = x2 - x1
        delta_y = y2 - y1
        delta_z = z2 - z1
        distance = np.sqrt(delta_x**2 + delta_y**2 + delta_z**2)
        
        # Prepare all 11 inputs for the CPPN
        input_coords = [
            x1, y1, z1,          # Source coordinates (3)
            x2, y2, z2,          # Target coordinates (3)
            delta_x, delta_y, delta_z,  # Coordinate differences (3)
            distance,            # Euclidean distance (1)
            1.0                  # Bias (1)
        ]
        
        output = cppn.activate(input_coords)[0]
        return output if abs(output) > self.weight_threshold else 0

    def create_phenotype_network(self, cppn, substrate):
        """
        Create neural network (phenotype) from CPPN and substrate
        """
        weights = {}
        
        # 1. Connect input directly to output as well
        for i, input_coord in enumerate(substrate.input_coords):
            if i >= 11:  # Skip after 11 inputs
                break
            for j, output_coord in enumerate(substrate.output_coords):
                weight = self.query_cppn(cppn, input_coord, output_coord)
                if weight != 0:
                    weights[(i, j + len(substrate.input_coords) + 
                           sum(len(coords) for coords in substrate.hidden_coords))] = weight

        # 2. Connect input to first hidden layer with proper indexing
        hidden_start_idx = len(substrate.input_coords)
        for i, input_coord in enumerate(substrate.input_coords):
            if i >= 11:  # Skip after 11 inputs
                break
            for j, hidden_coord in enumerate(substrate.hidden_coords[0]):
                weight = self.query_cppn(cppn, input_coord, hidden_coord)
                if weight != 0:
                    weights[(i, j + hidden_start_idx)] = weight

        # 3. Connect between hidden layers with correct indexing
        current_idx = hidden_start_idx
        for layer_idx in range(len(substrate.hidden_coords) - 1):
            current_coords = substrate.hidden_coords[layer_idx]
            next_coords = substrate.hidden_coords[layer_idx + 1]
            next_idx = current_idx + len(current_coords)
            
            for i, source_coord in enumerate(current_coords):
                for j, target_coord in enumerate(next_coords):
                    weight = self.query_cppn(cppn, source_coord, target_coord)
                    if weight != 0:
                        weights[(i + current_idx, j + next_idx)] = weight
            current_idx = next_idx

        # 4. Connect last hidden layer to output with correct indexing
        if substrate.hidden_coords:
            output_start_idx = hidden_start_idx + sum(len(coords) for coords in substrate.hidden_coords)
            last_hidden_coords = substrate.hidden_coords[-1]
            
            for i, hidden_coord in enumerate(last_hidden_coords):
                for j, output_coord in enumerate(substrate.output_coords):
                    weight = self.query_cppn(cppn, hidden_coord, output_coord)
                    if weight != 0:
                        weights[(i + current_idx, j + output_start_idx)] = weight

        return weights

    def _activate(self, x):
        """
        Safe sigmoid activation function that clips extreme values
        to prevent overflow
        """
        # Clip values to prevent overflow
        x = np.clip(x, -100, 100)
        return 1 / (1 + np.exp(-x))
    
    def _activate_relu(self, x):
        """ReLU activation function"""
        return np.maximum(0, x)

    def _activate_sigmoid(self, x):
        """Sigmoid activation function (for output layer)"""
        x = np.clip(x, -100, 100)
        return 1 / (1 + np.exp(-x))

    # def _forward_pass(self, inputs, weights):
    #     """
    #     Perform forward pass through the phenotype network
    #     Returns list of layer activations, with final layer being a single output
    #     between 0 and 1
    #     """
    #     # Calculate total nodes in network
    #     total_nodes = (len(self.substrate.input_coords) + 
    #                   sum(len(coords) for coords in self.substrate.hidden_coords) +
    #                   len(self.substrate.output_coords))
        
    #     # Initialize activations for all nodes
    #     all_activations = np.zeros(total_nodes)
    #     all_activations[:len(inputs)] = inputs  # Set input values
        
    #     # Process all connections with scaling factor to prevent explosion
    #     scale_factor = 1.0 / max(1, len(weights))  # Scale based on number of connections
    #     for (source, target), weight in weights.items():
    #         all_activations[target] += all_activations[source] * weight * scale_factor
        
    #     # Apply activation function to hidden and output nodes
    #     hidden_start = len(self.substrate.input_coords)
    #     all_activations[hidden_start:] = self._activate(all_activations[hidden_start:])
        
    #     # Extract layer-wise activations for return
    #     current_idx = 0
    #     layer_activations = [inputs]
        
    #     # Extract hidden layer activations
    #     for hidden_coords in self.substrate.hidden_coords:
    #         layer_size = len(hidden_coords)
    #         start_idx = current_idx + len(self.substrate.input_coords)
    #         layer_activations.append(all_activations[start_idx:start_idx + layer_size])
    #         current_idx += layer_size
        
    #     # Extract output layer activations (will be a single value between 0 and 1)
    #     output_size = len(self.substrate.output_coords)
    #     final_output = all_activations[-output_size:]
    #     layer_activations.append(final_output)
        
    #     return layer_activations

    def _forward_pass(self, inputs, weights):
        """
        Perform forward pass through the phenotype network.
        ReLU for hidden layers, sigmoid for the output layer.
        """
        total_nodes = (len(self.substrate.input_coords) + 
                    sum(len(coords) for coords in self.substrate.hidden_coords) + 
                    len(self.substrate.output_coords))

        # Initialize activations for all nodes
        all_activations = np.zeros(total_nodes)
        all_activations[:len(inputs)] = inputs  # Set input values
        
        scale_factor = 1.0 / max(1, len(weights))  # Scale to prevent explosion
        for (source, target), weight in weights.items():
            all_activations[target] += all_activations[source] * weight * scale_factor

        # Apply ReLU to all hidden layer activations
        hidden_start = len(self.substrate.input_coords)
        hidden_end = hidden_start + sum(len(coords) for coords in self.substrate.hidden_coords)
        all_activations[hidden_start:hidden_end] = self._activate_relu(all_activations[hidden_start:hidden_end])

        # Apply sigmoid to the output layer
        output_start = hidden_end
        all_activations[output_start:] = self._activate_sigmoid(all_activations[output_start:])

        # Extract layer-wise activations for return (optional if needed)
        layer_activations = [inputs]
        current_idx = len(inputs)

        # Hidden layers
        for hidden_coords in self.substrate.hidden_coords:
            layer_size = len(hidden_coords)
            layer_activations.append(all_activations[current_idx:current_idx + layer_size])
            current_idx += layer_size

        # Output layer
        output_activations = all_activations[-len(self.substrate.output_coords):]
        layer_activations.append(output_activations)

        return layer_activations

    def eval_genomes(self, genomes, config):
        """
        Evaluation function for CPPN genomes
        """
        self.env.populate_organisms(genomes)
        
        for genome_id, genome in genomes:
            cppn = neat.nn.FeedForwardNetwork.create(genome, config)
            weights = self.create_phenotype_network(cppn, self.substrate)
            
            y_true = []
            y_scores = []
            
            # Evaluate against all data rows in current batch
            for data_row in self.env.data_rows:
                # Forward pass through phenotype network
                activations = self._forward_pass(data_row.transformed_data, weights)
                output = activations[-1][0]  # Get single output value
                
                # Store prediction (threshold at 0.5 for binary classification)
                data_row.predicted_answer = 'Y' if output > 0.5 else 'N'
                y_true.append(1 if data_row.answer == 'Y' else 0)
                y_scores.append(output)
            
            try:
                # Calculate fitness using both AUC-ROC and accuracy
                auc_roc = roc_auc_score(y_true, y_scores)
                correct_predictions = sum(1 for true, score in zip(y_true, y_scores) 
                                    if (true == 1 and score > 0.5) or (true == 0 and score <= 0.5))
                accuracy = correct_predictions / len(y_true)
                genome.fitness = (auc_roc + accuracy) / 2
            except ValueError:
                # Fallback to just accuracy if AUC-ROC fails
                accuracy = sum(1 for true, score in zip(y_true, y_scores) 
                            if (true == 1 and score > 0.5) or (true == 0 and score <= 0.5)) / len(y_true)
                genome.fitness = accuracy
        
        # Load next batch for next generation
        self.env.load_next_batch()

    def _activate(self, x):
        """Activation function (sigmoid)"""
        return 1 / (1 + np.exp(-x))


    def _visualize_results(self, winner):
        """
        Visualize the CPPN and evolution statistics
        """
        feature_names = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 
                'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 
                'Loan_Amount_Term', 'Credit_History', 'Property_Area']

    
        node_names = {-(i+1): name for i, name in enumerate(feature_names)}
        node_names[0] = 'output' 

        # try:
        visualize.draw_net(self.config, winner, view=True, node_names=node_names,
                            filename=os.path.join(self.output_dir, 'winner_cppn.svg'))
        visualize.plot_stats(self.stats, ylog=False, view=True,
                            filename=os.path.join(self.output_dir, 'fitness_history.svg'))
        visualize.plot_species(self.stats, view=True,
                            filename=os.path.join(self.output_dir, 'speciation.svg'))
        # except:
        #     pass

if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-cppn.txt') 
    dataset_path = r"c:\Users\engin\OneDrive\Desktop\DATA201\Project\train-1.csv"
    
    hyper_neat = HyperNEAT(config_path, dataset_path)
    winner = hyper_neat.run()