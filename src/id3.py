from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
import pandas as pd

class ID3Regressor:
    def __init__(self, data_instance, target_column, feature_columns, max_depth=5, test_data=None):
        self.data_instance = data_instance
        self.target_column = target_column
        self.feature_columns = feature_columns
        self.max_depth = max_depth
        self.tree_model = None
        self.test_data = test_data  # Test data can be provided externally

    def prepare_id3_data(self, year=None, test_month=12):
        if self.test_data is not None:
            # Use the external test data provided
            train_data = self.data_instance.data[self.data_instance.data['Year'] != year]
            test_data = self.test_data[self.test_data['Year'] == year]
        else:
            # Use internal data filtering
            train_data, test_data = self.data_instance.get_train_test_split(year, test_month)

        X_train = train_data[self.feature_columns]
        y_train = train_data[self.target_column]

        X_test = test_data[self.feature_columns]
        y_test = test_data[self.target_column]

        return X_train, y_train, X_test, y_test

    def train(self, year=None, test_month=12):
        X_train, y_train, X_test, y_test = self.prepare_id3_data(year, test_month)

        # Debugging: Print dataset shapes
        print(f"Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")

        # Check if the test set is empty
        if X_test.shape[0] == 0:
            raise ValueError(f"No test data available for year={year} and test_month={test_month}. Check your dataset.")

        self.tree_model = DecisionTreeRegressor(max_depth=self.max_depth)
        self.tree_model.fit(X_train, y_train)

        predictions = self.tree_model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        print(f"MAE for ID3 (year={year}, test_month={test_month}): {mae}")

        return predictions, mae

    def predict(self, X):
        if self.tree_model is None:
            raise ValueError("The model has not been trained yet. Call the 'train' method first.")
        return self.tree_model.predict(X)

    def feature_importances(self):
        if self.tree_model is None:
            raise ValueError("The model has not been trained yet. Call the 'train' method first.")
        importances = self.tree_model.feature_importances_
        return dict(zip(self.feature_columns, importances))

    @staticmethod
    def bucketize_target(data, n_bins=10):
        if 'Sold[MW]' not in data.columns:
            raise ValueError("Column 'Sold[MW]' is missing from the dataset.")

        # Apply bucketing
        result = data.copy()
        result['Sold_bucket'] = pd.cut(
            result['Sold[MW]'],
            bins=n_bins,
            labels=False,
            include_lowest=True
        )
        print("Discretized 'Sold[MW]' into 'Sold_bucket' with", n_bins, "bins.")
        return result
