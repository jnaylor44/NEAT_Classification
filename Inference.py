import neat
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import os
import warnings
from collections import Counter
from Gitpost import Environment, DataPreprocessor, OutlierRemover
from sklearn.base import BaseEstimator, TransformerMixin

def run_inference(winner_path, preprocessor_path, config_path, test_data_path):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    with open(winner_path, 'rb') as f:
        winner = pickle.load(f)
    with open(preprocessor_path, 'rb') as f:
        preprocessor = pickle.load(f)
    
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    
    test_data = pd.read_csv(test_data_path)
    if 'Loan_ID' in test_data.columns:
        test_data = test_data.drop('Loan_ID', axis=1)
        test_data = test_data.dropna()
    
    y_true = test_data['Loan_Status'].values if 'Loan_Status' in test_data.columns else None
    X_test = test_data.drop('Loan_Status', axis=1) if 'Loan_Status' in test_data.columns else test_data

    y_pred = []
    raw_outputs = []
    
    for idx in range(len(X_test)):
        row = X_test.iloc[idx]
        transformed_row = preprocessor.transform_row(row)
        output = winner_net.activate(transformed_row)[0]
        raw_outputs.append(output)
        prediction = 'Y' if output > 0.5 else 'N'
        y_pred.append(prediction)

    if y_true is not None:
        pred_counter = Counter(y_pred)
        true_counter = Counter(y_true)
        print("\nPrediction Distribution:")
        print(f"Predicted: {dict(pred_counter)}")
        print(f"Actual: {dict(true_counter)}")
        
        y_true_binary = np.where(y_true == 'Y', 1, 0)
        auc_roc = roc_auc_score(y_true_binary, raw_outputs)
        accuracy = accuracy_score(y_true, y_pred)
        
        print(f"\nPerformance Metrics:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"AUC-ROC: {auc_roc:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_true, y_pred))
    
    return y_pred, raw_outputs

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    winner_path = os.path.join(local_dir, "neat_outputs", "winner.pkl")
    preprocessor_path = os.path.join(local_dir, "neat_outputs", "preprocessor.pkl")
    config_path = os.path.join(local_dir, "config-feedforward")
    test_data_path = r"c:\Users\engin\OneDrive\Desktop\DATA201\Project\test-1.csv"
    
    predictions, raw_outputs = run_inference(winner_path, preprocessor_path, config_path, test_data_path)

