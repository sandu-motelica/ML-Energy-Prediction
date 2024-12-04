import pandas as pd
from sklearn.model_selection import train_test_split
from id3 import ID3Regressor
from bayes import BayesianRegressor
from evaluate import evaluate_model


def train_and_evaluate_models(data_path, features, target, test_size=0.2, random_state=42):
    data = pd.read_csv(data_path)
    X = data[features]
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    results = {}

    id3_model = ID3Regressor(max_depth=5, bucket_size=10)
    id3_model.fit(X_train, y_train)
    id3_predictions = id3_model.predict(X_test)
    results['ID3'] = evaluate_model(y_test, id3_predictions, model_name="ID3 Regressor")

    bayes_model = BayesianRegressor(bucket_size=10)
    bayes_model.fit(X_train, y_train)
    bayes_predictions = bayes_model.predict(X_test)
    results['Bayesian'] = evaluate_model(y_test, bayes_predictions, model_name="Bayesian Regressor")

    return results
