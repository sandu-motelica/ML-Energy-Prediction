import numpy as np
from sklearn.preprocessing import KBinsDiscretizer

class BayesianRegressor:
    """
    Implements a naive Bayesian approach for regression using discretized features.
    """
    def __init__(self, data, feature_columns, target_column, test_data=None, n_bins=10):
        self.data = data
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.test_data = test_data.copy() if test_data is not None else None
        self.n_bins = n_bins
        self.probabilities = None

    def preprocess_data(self):
        """
        Discretizes continuous features into bins for Bayesian computation.
        """
        discretizer = KBinsDiscretizer(n_bins=self.n_bins, encode='ordinal', strategy='uniform')
        self.data.loc[:, self.feature_columns] = discretizer.fit_transform(self.data[self.feature_columns])
        if self.test_data is not None:
            self.test_data.loc[:, self.feature_columns] = discretizer.transform(self.test_data[self.feature_columns])

    def calculate_probabilities(self):
        probabilities = {}
        grouped = self.data.groupby(self.feature_columns)[self.target_column]
        for group, values in grouped:
            probabilities[group] = values.mean()
        self.probabilities = probabilities

    def predict(self, test_data):
        predictions = []
        for _, row in test_data.iterrows():
            features = tuple(row[self.feature_columns])
            prediction = self.probabilities.get(features, np.mean(self.data[self.target_column]))
            predictions.append(prediction)
        return np.array(predictions)

    def train(self):
        """
        Trains the Bayesian model by calculating probabilities for discretized features.
        """
        self.preprocess_data()
        self.calculate_probabilities()
        predictions = self.predict(self.test_data)

        return predictions
