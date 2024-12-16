class DataRow:
    def __init__(self, raw_data, transformed_data, index, answer):
        self.raw_data = raw_data
        self.transformed_data = transformed_data
        self.index = index
        self.answer = answer
        self.organism = None
        self.predicted_answer = None
        self.show_prediction = False
        