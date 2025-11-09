class ClassificationTrainer:
    def __init__(self, scripts_path, output_path):
        self.scripts_path = scripts_path
        self.output_path = output_path

    def _load_models(self):
        # same as RegressionTrainer, but filter for "classification"
        ...

    def train_all(self, X_train, y_train, X_val, y_val):
        # same logic
        ...
