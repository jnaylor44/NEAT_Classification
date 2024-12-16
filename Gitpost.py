import random
import numpy as np
import neat
import os
import copy
import pandas as pd
import visualize
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator,TransformerMixin
import pickle

from Environment import Environment
from Preprocessor import DataPreprocessor, OutlierRemover
from Organism import Organism
from DataRow import DataRow
    
def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        y_true = []
        y_scores = []

        for data_row in env.data_rows:
            output = net.activate(data_row.transformed_data)[0]
            y_true.append(1 if data_row.answer == 'Y' else 0)
            y_scores.append(output)

        try:
            auc_roc = roc_auc_score(y_true, y_scores)
            correct_predictions = sum(1 for true, score in zip(y_true, y_scores) if (true == 1 and score > 0.5) or (true == 0 and score <= 0.5))
            accuracy = correct_predictions / len(y_true)
            genome.fitness = (auc_roc+accuracy)/2
        except ValueError:
            accuracy = sum(1 for true, score in zip(y_true, y_scores) if (true == 1 and score > 0.5) or (true == 0 and score <= 0.5))/len(y_true)
            genome.fitness = accuracy

    env.load_next_batch()

def run_neat(config_path, dataset_path):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        config_path)

    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(50))

    global env
    env = Environment(config, dataset_path)

    winner = p.run(eval_genomes, 64)

    print("\nBest genome:\n{!s}".format(winner))

    output_dir = "neat_outputs"
    os.makedirs(output_dir, exist_ok=True)
    winner_path = os.path.join(output_dir, "winner.pkl")
    preprocessor_path = os.path.join(output_dir, "preprocessor.pkl")
    with open(winner_path, "wb") as f:
        pickle.dump(winner, f)
    with open(preprocessor_path, "wb") as f:
        pickle.dump(env.preprocessor, f)
    
    print(f"Saved winner genome to {winner_path}")
    print(f"Saved preprocessor to {preprocessor_path}")

    feature_names = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 
                    'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 
                    'Loan_Amount_Term', 'Credit_History', 'Property_Area']

    
    node_names = {-(i+1): name for i, name in enumerate(feature_names)}
    node_names[0] = 'output' 

    visualize.draw_net(config, winner, view=True, node_names=node_names,
                    filename=os.path.join(output_dir, 'winner_network.svg'))
    visualize.plot_stats(stats, ylog=False, view=True,
                        filename=os.path.join(output_dir, 'fitness_history.svg'))
    visualize.plot_species(stats, view=True,
                        filename=os.path.join(output_dir, 'speciation.svg'))

    return winner

if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    dataset_path = r'c:\Users\engin\OneDrive\Desktop\DATA201\Project\train-1.csv'
    
    run_neat(config_path, dataset_path)