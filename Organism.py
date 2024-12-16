import neat

class Organism:
    def __init__(self, genome, config, data_row):
        self.genome = genome
        self.net = neat.nn.FeedForwardNetwork.create(genome, config)
        self.data_row = data_row
        self.prediction = None
        self.is_correct = None
        self.show_result = False
        self.output_value = None 

    def make_prediction(self):
        if self.prediction is None:
            self.output_value = self.net.activate(self.data_row.transformed_data)[0]
            self.prediction = 'Y' if self.output_value > 0.5 else 'N'
            self.data_row.predicted_answer = self.prediction
            self.is_correct = self.prediction == self.data_row.answer