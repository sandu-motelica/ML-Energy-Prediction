from sklearn.metrics import root_mean_squared_error, mean_absolute_error

def calculate_rmse(y_true, y_pred):
    return root_mean_squared_error(y_true, y_pred)
def calculate_mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

def evaluate_model(y_true, y_pred, model_name="Model"):
    rmse = calculate_rmse(y_true, y_pred)
    mae = calculate_mae(y_true, y_pred)
    print(f"{model_name} Performance:")
    print(f" - RMSE: {rmse}")
    print(f" - MAE: {mae}")
    return {"RMSE": rmse, "MAE": mae}
