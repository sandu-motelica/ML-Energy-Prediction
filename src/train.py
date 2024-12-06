import pandas as pd
from id3 import ID3Regressor
from bayes import BayesianRegressor
from evaluate import evaluate_model

def direct_prediction(data, test_data, feature_columns, target_column, depth=8):
    print("=== Direct Prediction of `Sold[MW]` ===")
    id3_model = ID3Regressor(
        data=data,
        target_column=target_column,
        feature_columns=feature_columns,
        max_depth=depth,
        test_data=test_data
    )

    try:
        predictions = id3_model.train(year=2023, test_month=12)
        evaluate_model(predictions, test_data[target_column], "Direct Prediction")
    except ValueError as e:
        print(f"Error during training or evaluation: {e}")

    return id3_model


def component_prediction(data, test_data, feature_columns, depth=8, year=2023, month=12):
    print("=== Predict Components (`Consum[MW]` and `Productie[MW]`) ===")

    consum_model = ID3Regressor(
        data=data,
        target_column='Consum[MW]',
        feature_columns=feature_columns,
        max_depth=depth,
        test_data=test_data
    )
    productie_model = ID3Regressor(
        data=data,
        target_column='Productie[MW]',
        feature_columns=feature_columns,
        max_depth=depth,
        test_data=test_data
    )

    consum_predictions= consum_model.train(year=year, test_month=month)
    productie_predictions = productie_model.train(year=year, test_month=month)
    sold_predictions = productie_predictions - consum_predictions

    evaluate_model(sold_predictions, test_data['Sold[MW]'], "Component Prediction (Productie - Consum)")

    return consum_model, productie_model


def bucket_prediction(data, test_data, feature_columns, n_bins=10, depth=8, year=2023, month=12):
    print("=== Bucketing `Sold[MW]` ===")
    data = ID3Regressor.bucketize_target(data, n_bins=n_bins)
    test_data = ID3Regressor.bucketize_target(test_data, n_bins=n_bins)

    id3_model_bucket = ID3Regressor(
        data=data,
        target_column='Sold_bucket',
        feature_columns=feature_columns,
        max_depth=depth,
        test_data=test_data
    )

    bucket_predictions = id3_model_bucket.train(year=year, test_month=month)

    bucket_intervals = pd.IntervalIndex.from_breaks(
        pd.cut(data['Sold[MW]'], bins=n_bins, retbins=True)[1]
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

    evaluate_model(continuous_predictions, test_data['Sold[MW]'], "Bucketing `Sold[MW]`")

    return id3_model_bucket
