from sklearn.tree import DecisionTreeRegressor
import pandas as pd
from preprocess import Data
from sklearn.model_selection import GridSearchCV


class ID3Regressor:
    """
    Adapts DecisionTreeRegressor for regression tasks, supporting feature importance and bucketing.
    """
    def __init__(self, data, target_column, feature_columns, max_depth=5, test_data=None):
        self.data = data
        self.target_column = target_column
        self.feature_columns = feature_columns
        self.max_depth = max_depth
        self.tree_model = None
        self.test_data = test_data

    def prepare_id3_data(self, year=None, test_month=12):
        """
        Prepares training and testing data split by month and year.
        """
        if self.test_data is not None:
            train_data = self.data[self.data['Data'].dt.month != test_month]
            test_data = self.test_data[self.test_data['Data'].dt.year == year]
        else:
            train_data, test_data = Data.get_train_test_split(self.data, year, test_month)

        X_train = train_data[self.feature_columns]
        y_train = train_data[self.target_column]

        X_test = test_data[self.feature_columns]
        y_test = test_data[self.target_column]

        return X_train, y_train, X_test, y_test

    def train(self, year=None, test_month=12):
        """
        Trains the model using provided data, splitting by month for testing.
        """
        X_train, y_train, X_test, y_test = self.prepare_id3_data(year, test_month)

        if X_test.shape[0] == 0:
            raise ValueError(f"No test data available for year={year} and test_month={test_month}. Check your dataset.")

        # parameters = self.optimize_id3_hyperparameters()
        # self.tree_model = DecisionTreeRegressor(
        #     max_depth=parameters['max_depth'],
        #     min_samples_leaf=parameters['min_samples_leaf'],
        #     min_samples_split=parameters['min_samples_split'])
        self.tree_model = DecisionTreeRegressor(
            max_depth=self.max_depth,)
        self.tree_model.fit(X_train, y_train)

        predictions = self.tree_model.predict(X_test)

        return predictions

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

        result = data.copy()
        result['Sold_bucket'] = pd.cut(
            result['Sold[MW]'],
            bins=n_bins,
            labels=False,
            include_lowest=True
        )
        return result

    def optimize_id3_hyperparameters(self):
        """
        Optimizes the hyperparameters of the ID3 algorithm using GridSearchCV.
        """
        print("=== Optimizing ID3 Hyperparameters ===")
        X_train = self.data[self.feature_columns]
        y_train = self.data[self.target_column]

        param_grid = {
            'max_depth': [3, 5, 10, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }

        grid_search = GridSearchCV(DecisionTreeRegressor(), param_grid, cv=3, scoring='neg_mean_absolute_error')
        grid_search.fit(X_train, y_train)

        print(f"Best Parameters: {grid_search.best_params_}")
        return grid_search.best_params_
