from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

def calculate_rmse(y_true, y_pred):
    return root_mean_squared_error(y_true, y_pred)

def calculate_mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

def evaluate_model(predictions, y_test, approach_name):
    mae = mean_absolute_error(y_test, predictions)
    rmse = root_mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"Results for {approach_name}:")
    print(f"  MAE: {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  R²: {r2:.2f}\n")
