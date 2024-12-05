import pandas as pd
from sklearn.model_selection import train_test_split
from id3 import ID3Regressor
from bayes import BayesianRegressor
from evaluate import evaluate_model

def direct_prediction(data_handler, test_data, feature_columns, target_column):
    print("=== Direct Prediction of `Sold[MW]` ===")
    id3_model = ID3Regressor(
        data_instance=data_handler,
        target_column=target_column,
        feature_columns=feature_columns,
        max_depth=8,
        test_data=test_data
    )

    try:
        predictions, _ = id3_model.train(year=2023, test_month=12)
        evaluate_model(predictions, test_data[target_column], "Direct Prediction")
    except ValueError as e:
        print(f"Error during training or evaluation: {e}")

    return id3_model


def component_prediction(data_handler, test_data, feature_columns):
    print("=== Predict Components (`Consum[MW]` and `Productie[MW]`) ===")

    consum_model = ID3Regressor(
        data_instance=data_handler,
        target_column='Consum[MW]',
        feature_columns=feature_columns,
        max_depth=8,
        test_data=test_data
    )
    productie_model = ID3Regressor(
        data_instance=data_handler,
        target_column='Productie[MW]',
        feature_columns=feature_columns,
        max_depth=8,
        test_data=test_data
    )

    consum_predictions, _ = consum_model.train(year=2023, test_month=12)
    productie_predictions, _ = productie_model.train(year=2023, test_month=12)
    sold_predictions = productie_predictions - consum_predictions

    evaluate_model(sold_predictions, test_data['Sold[MW]'], "Component Prediction (Productie - Consum)")

    return consum_model, productie_model


def bucket_prediction(data_handler, test_data, feature_columns):
    print("=== Bucketing `Sold[MW]` ===")
    my_n_bins = 10
    data_handler.data = ID3Regressor.bucketize_target(data_handler.data, n_bins=my_n_bins)
    test_data = ID3Regressor.bucketize_target(test_data, n_bins=my_n_bins)

    id3_model_bucket = ID3Regressor(
        data_instance=data_handler,
        target_column='Sold_bucket',
        feature_columns=feature_columns,
        max_depth=8,
        test_data=test_data
    )

    bucket_predictions, _ = id3_model_bucket.train(year=2023, test_month=12)

    bucket_intervals = pd.IntervalIndex.from_breaks(
        pd.cut(data_handler.data['Sold[MW]'], bins=my_n_bins, retbins=True)[1]
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
