import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.preprocess import Data
from src.train import *

def preprocess_data():
    excel_folder = os.path.join("data", "raw_data")
    excel_files = [
        # os.path.join(excel_folder, "2021.xlsx"),
        os.path.join(excel_folder, "2022.xlsx"),
        os.path.join(excel_folder, "2023.xlsx"),
        os.path.join(excel_folder, "2024.xlsx")
    ]
    data_handler = Data(excel_files)
    data_handler.process_and_save_csv(add_time_attributes=True, add_production_attributes=True)

def summarize_results(all_results):
    metrics = ["mae", "rmse", "r2"]
    best_overall = {metric: None for metric in metrics}

    print("\n=== Summary of Results ===")
    for result in all_results:
        print(f"Feature Set: {result['features']}")
        for model_result in result['results']:
            print(f"  {model_result['name']}: MAE={model_result['mae']:.4f}, RMSE={model_result['rmse']:.4f}, R²={model_result['r2']:.4f}")

        for metric in metrics:
            best_model = min(result['results'], key=lambda x: x[metric]) if metric != "r2" else max(result['results'], key=lambda x: x[metric])
            if not best_overall[metric] or (
                best_overall[metric][metric] > best_model[metric] if metric != "r2" else best_overall[metric][metric] < best_model[metric]
            ):
                best_overall[metric] = best_model

    print("\n=== Best Models by Metric ===")
    for metric, model in best_overall.items():
        print(f"Best Model by {metric.upper()}: {model['name']} with {metric.upper()}={model[metric]:.4f}")
    print("\n")

def main():
    # preprocess_data()

    data_2023 = Data.load_csv("data/processed_data/2023.csv")
    data_train, data_test = Data.get_train_test_split(data_2023, 2023)

    feature_sets = [
        ['Day_of_Week', 'Hour', 'Consum[MW]', 'Intermittent_Production', 'Constant_Production'],
        ['Day_of_Week', 'Hour', 'Consum[MW]', 'Productie[MW]'],
        ['Hour', 'Consum[MW]', 'Productie[MW]'],
        ['Consum[MW]', 'Productie[MW]']
    ]
    target_column = 'Sold[MW]'

    all_results = []

    for feature_columns in feature_sets:
        result = train_with_all_classifiers(data_train, data_test, feature_columns, target_column)
        all_results.append(result)

    summarize_results(all_results)

    data_train_all_years = Data.load_csv("data/processed_data/all_years.csv")
    data_test_all_years = Data.load_csv("data/processed_data/all_december.csv")
    _, data_test_2023 = Data.get_train_test_split(data_test_all_years, 2023)

    all_results = []

    for feature_columns in feature_sets:
        result = train_with_all_classifiers(data_train_all_years, data_test_2023, feature_columns, target_column)
        all_results.append(result)

    summarize_results(all_results)


if __name__ == "__main__":
    main()