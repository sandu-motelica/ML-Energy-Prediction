import pandas as pd
from id3 import ID3Regressor
from bayes import BayesianRegressor
from evaluate import evaluate_model


def direct_prediction(data_train, data_test, feature_columns, target_column, depth=8):
    print("=== ID3 Direct Prediction of `Sold[MW]` ===")
    data_train = data_train.copy()
    data_test = data_test.copy()

    id3_model = ID3Regressor(
        data=data_train,
        target_column=target_column,
        feature_columns=feature_columns,
        max_depth=depth,
        test_data=data_test
    )

    predictions = id3_model.train(year=2023, test_month=12)
    mae, rmse, r2 = evaluate_model(predictions, data_test[target_column], "Direct Prediction")
    importances = id3_model.feature_importances()
    return {"name": "Direct Prediction", "predictions": predictions, "mae": mae, "rmse": rmse, "r2": r2, "importances": importances}



def component_prediction(data_train, data_test, feature_columns, depth=8, year=2023, month=12):
    print("=== ID3 Predict Components (`Consum[MW]` and `Productie[MW]`) ===")
    data_train = data_train.copy()
    data_test = data_test.copy()

    feature_cols_consum = list(set(feature_columns).difference(['Consum[MW]']))
    consum_model = ID3Regressor(
        data=data_train,
        target_column='Consum[MW]',
        feature_columns=feature_cols_consum,
        max_depth=depth,
        test_data=data_test
    )
    feature_cols_productie = list(set(feature_columns).difference(['Productie[MW]']))
    productie_model = ID3Regressor(
        data=data_train,
        target_column='Productie[MW]',
        feature_columns=feature_cols_productie,
        max_depth=depth,
        test_data=data_test
    )

    consum_predictions= consum_model.train(year=year, test_month=month)
    productie_predictions = productie_model.train(year=year, test_month=month)
    sold_predictions = productie_predictions - consum_predictions

    mae, rmse, r2 = evaluate_model(sold_predictions, data_test['Sold[MW]'], "Component Prediction (Productie - Consum)")
    importances_consum = consum_model.feature_importances()
    importances_productie = productie_model.feature_importances()

    return {"name": "Component Prediction", "predictions": sold_predictions, "mae": mae, "rmse": rmse, "r2": r2, "importances_consum": importances_consum, "importances_productie": importances_productie}



def bucket_prediction(data_train, data_test, feature_columns, n_bins=10, depth=8, year=2023, month=12):
    print("=== ID3 Bucketing `Sold[MW]` ===")
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

    bucket_intervals = pd.IntervalIndex.from_breaks(pd.cut(data_train['Sold[MW]'], bins=n_bins, retbins=True)[1])

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


    mae, rmse, r2 = evaluate_model(continuous_predictions, data_test['Sold[MW]'], "Bucketing `Sold[MW]`")
    importances = id3_model_bucket.feature_importances()

    return {"name": "Bucket Prediction", "predictions": continuous_predictions, "mae": mae, "rmse": rmse, "r2": r2, "importances": importances}


def bayesian_prediction(data_train, data_test, feature_columns, target_column, n_bins=10):
    print("=== Bayesian Regression ===")
    data_train = data_train.copy()
    data_test = data_test.copy()

    bayesian_model = BayesianRegressor(
        data=data_train,
        feature_columns=feature_columns,
        target_column=target_column,
        test_data=data_test,
        n_bins=n_bins)

    predictions = bayesian_model.train()
    mae, rmse, r2 = evaluate_model(predictions, data_test[target_column], "Bayesian Regression")
    return {"name": "Bayesian Prediction", "predictions": predictions, "mae": mae, "rmse": rmse, "r2": r2}



def train_with_all_classifiers(data_train, data_test, feature_columns, target_column):
    print(f"Train with {feature_columns}")
    print(f"Train set shape: {data_train.shape}, Test set shape: {data_test.shape}")

    results = []
    results.append(bayesian_prediction(data_train, data_test, feature_columns, target_column))
    results.append(direct_prediction(data_train, data_test, feature_columns, target_column))
    results.append(component_prediction(data_train, data_test, feature_columns, year=2023, month=12))
    results.append(bucket_prediction(data_train, data_test, feature_columns, n_bins=10))

    print("=== Feature Importances ===")
    print("Direct Model Feature Importances:", results[1]["importances"])
    print("Consum Model Feature Importances:", results[2]["importances_consum"])
    print("Productie Model Feature Importances:", results[2]["importances_productie"])
    print("Bucket Model Feature Importances:", results[3]["importances"])
    print("\n")

    return {"results": results, "features": feature_columns}
