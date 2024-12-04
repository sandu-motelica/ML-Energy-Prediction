import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

class ID3Regressor:
    def __init__(self, max_depth=None, bucket_size=10):
        self.max_depth = max_depth
        self.bucket_size = bucket_size
        self.model = None

    def discretize_target(self, y):
        bins = np.linspace(y.min(), y.max(), self.bucket_size)
        labels = range(len(bins) - 1)
        y_discretized = pd.cut(y, bins=bins, labels=labels, include_lowest=True)
        return y_discretized.astype(int)

    def fit(self, X, y):
        y_discretized = self.discretize_target(y)
        self.model = DecisionTreeClassifier(max_depth=self.max_depth)
        self.model.fit(X, y_discretized)
        print("ID3 Model trained successfully.")

    def predict(self, X):
        predictions = self.model.predict(X)
        bins = np.linspace(0, 1, self.bucket_size)
        return bins[predictions]

    def predict_bucket(self, X):
        return self.model.predict(X)
