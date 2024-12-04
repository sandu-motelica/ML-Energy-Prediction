import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB

class BayesianRegressor:
    def __init__(self, bucket_size=10):
        self.bucket_size = bucket_size
        self.model = GaussianNB()

    def discretize_target(self, y):
        bins = np.linspace(y.min(), y.max(), self.bucket_size)
        labels = range(len(bins) - 1)
        y_discretized = pd.cut(y, bins=bins, labels=labels, include_lowest=True)
        return y_discretized.astype(int), bins

    def fit(self, X, y):
        y_discretized, self.bins = self.discretize_target(y)
        self.model.fit(X, y_discretized)
        print("Bayesian Model trained successfully.")

    def predict(self, X):
        predictions = self.model.predict(X)
        centers = (self.bins[:-1] + self.bins[1:]) / 2
        return centers[predictions]
