import sys
import os
import csv

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.preprocess import Data
from src.train import *

def preprocess_data():
    """
    Preprocesses raw data from Excel files, adds time and production attributes, and saves processed CSV files.
    """
    excel_folder = os.path.join("data", "raw_data")
    excel_files = [
        os.path.join(excel_folder, "2021.xlsx"),
        os.path.join(excel_folder, "2022.xlsx"),
        os.path.join(excel_folder, "2023.xlsx"),
        os.path.join(excel_folder, "2024.xlsx")
    ]
    data_handler = Data(excel_files)
    data_handler.process_and_save_csv(add_time_attributes=True, add_production_attributes=True)

def save_results_to_csv(all_results, file_path="results.csv"):
    """
    Saves the experiment results into a CSV file for further analysis.
    """
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Feature Set", "Model", "MAE", "RMSE", "R2"])
        for result in all_results:
            feature_set = result['features']
            for model_result in result['results']:
                writer.writerow([
                    feature_set, model_result['name'], model_result['mae'], model_result['rmse'], model_result['r2']
                ])
    print(f"Results saved to {file_path}")

def summarize_results(all_results):
    """
    Displays a summary of results for all experiments and identifies the best model per metric.
    """
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
    preprocess_data()

    # Load data for 2023 and split into training and testing sets
    data_2023 = Data.load_csv("data/processed_data/2023.csv")
    data_train, data_test = Data.get_train_test_split(data_2023, 2023)

    feature_sets = [
        ['Day_of_Week', 'Hour'],
        ['Day_of_Week', 'Hour', 'Consum[MW]', 'Intermittent_Production', 'Constant_Production'],
        ['Day_of_Week', 'Hour', 'Consum[MW]', 'Productie[MW]'],
        ['Consum[MW]', 'Productie[MW]']
    ]
    target_column = 'Sold[MW]'

    all_results = []

    # Train models using different feature sets
    for feature_columns in feature_sets:
        result = train_with_all_classifiers(data_train, data_test, feature_columns, target_column)
        all_results.append(result)

    summarize_results(all_results)
    save_results_to_csv(all_results, file_path="reports/results_2023_old.csv")

    # Train using all available years of data
    data_train_all_years = Data.load_csv("data/processed_data/all_years.csv")

    all_results = []

    for feature_columns in feature_sets:
        result = train_with_all_classifiers(data_train_all_years, data_test, feature_columns, target_column)
        all_results.append(result)

    summarize_results(all_results)
    save_results_to_csv(all_results, file_path="reports/results_all_years_old.csv")

if __name__ == "__main__":
    main()
