import pandas as pd
import random
from Preprocessor import DataPreprocessor
from DataRow import DataRow
from Organism import Organism

class Environment:
    def __init__(self, config, dataset_path):
        self.config = config
        self.full_dataset = pd.read_csv(dataset_path)
        self.preprocessor = None
        self.data_rows = []
        self.organisms = []
        self.generation = 0
        self.header = None 
        self.prepare_dataset()
        self.load_next_batch()
        

    def prepare_dataset(self):
        df = self.full_dataset.copy()
        if 'Loan_ID' in df.columns:
            df = df.drop('Loan_ID', axis=1)
        
        self.target_column = 'Loan_Status'
        self.labels = df[self.target_column].values
        self.feature_df = df.drop(self.target_column, axis=1)
        
        self.preprocessor = DataPreprocessor(self.feature_df)
        self.header = list(self.feature_df.columns) 

    def load_next_batch(self):
        self.data_rows = []
        random_indices = random.sample(range(len(self.feature_df)), 64)
        
        for i, idx in enumerate(random_indices):
            raw_data = self.feature_df.iloc[idx]
            transformed_data = self.preprocessor.transform_row(raw_data)
            self.data_rows.append(DataRow(raw_data, transformed_data, i, self.labels[idx]))

    def populate_organisms(self, genomes):
        self.organisms = []
        for i, (_, genome) in enumerate(genomes):
            if i < len(self.data_rows):
                organism = Organism(genome, self.config, self.data_rows[i])
                self.organisms.append(organism)
                self.data_rows[i].organism = organism 

    def show_results(self):
        if self.evaluation_complete:
            for organism in self.organisms:
                organism.show_result = True
                organism.data_row.show_prediction = True
            return False 
        return False