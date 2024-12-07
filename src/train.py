import pandas as pd
from id3 import ID3Regressor
from bayes import BayesianRegressor
from evaluate import evaluate_model


def direct_prediction(data_train, data_test, feature_columns, target_column, depth=8):
    print("=== Direct Prediction of `Sold[MW]` ===")
    data_train = data_train.copy()
    data_test = data_test.copy()

    id3_model = ID3Regressor(
        data=data_train,
        target_column=target_column,
        feature_columns=feature_columns,
        max_depth=depth,
        test_data=data_test
    )

    try:
        predictions = id3_model.train(year=2023, test_month=12)
        evaluate_model(predictions, data_test[target_column], "Direct Prediction")
    except ValueError as e:
        print(f"Error during training or evaluation: {e}")

    return id3_model


def component_prediction(data_train, data_test, feature_columns, depth=8, year=2023, month=12):
    print("=== Predict Components (`Consum[MW]` and `Productie[MW]`) ===")
    data_train = data_train.copy()
    data_test = data_test.copy()

    consum_model = ID3Regressor(
        data=data_train,
        target_column='Consum[MW]',
        feature_columns=feature_columns,
        max_depth=depth,
        test_data=data_test
    )
    productie_model = ID3Regressor(
        data=data_train,
        target_column='Productie[MW]',
        feature_columns=feature_columns,
        max_depth=depth,
        test_data=data_test
    )

    consum_predictions= consum_model.train(year=year, test_month=month)
    productie_predictions = productie_model.train(year=year, test_month=month)
    sold_predictions = productie_predictions - consum_predictions

    evaluate_model(sold_predictions, data_test['Sold[MW]'], "Component Prediction (Productie - Consum)")

    return consum_model, productie_model


def bucket_prediction(data_train, data_test, feature_columns, n_bins=10, depth=8, year=2023, month=12):
    print("=== Bucketing `Sold[MW]` ===")
    data_train = data_train.copy()
    data_test = data_test.copy()
    data_train = ID3Regressor.bucketize_target(data_train, n_bins=n_bins)
    data_test = ID3Regressor.bucketize_target(data_test, n_bins=n_bins)

    id3_model_bucket = ID3Regressor(
        data=data_train,
        target_column='Sold_bucket',
        feature_columns=feature_columns,
        max_depth=depth,
        test_data=data_test
    )

    bucket_predictions = id3_model_bucket.train(year=year, test_month=month)

    bucket_intervals = pd.IntervalIndex.from_breaks(
        pd.cut(data_train['Sold[MW]'], bins=n_bins, retbins=True)[1]
    )

    continuous_predictions = []
    for b in bucket_predictions:
        if pd.isna(b):
            continuous_predictions.append(0)
        else:
            b = int(b)
            if 0 <= b < len(bucket_intervals):
                continuous_predictions.append(bucket_intervals[b].mid)
            else:
                continuous_predictions.append(0)

    evaluate_model(continuous_predictions, data_test['Sold[MW]'], "Bucketing `Sold[MW]`")

    return id3_model_bucket


def bayesian_prediction(data_train, data_test, feature_columns, target_column, n_bins=10):
    print("=== Bayesian Regression ===")
    data_train = data_train.copy()
    data_test = data_test.copy()

    bayesian_model = BayesianRegressor(
        data=data_train,
        feature_columns=feature_columns,
        target_column=target_column,
        test_data=data_test,
        n_bins=n_bins
    )
    bayesian_model.train_and_evaluate()
