import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.preprocess import Data
from src.train import *

def preprocess_data():
    excel_folder = os.path.join("data", "raw_data")
    excel_files = [
        os.path.join(excel_folder, "2022.xlsx"),
        os.path.join(excel_folder, "2023.xlsx"),
        os.path.join(excel_folder, "2024.xlsx")
    ]
    data_handler = Data(excel_files)
    data_handler.process_and_save_csv(add_time_attributes=True, add_production_attributes=True)
    # data_handler = Data()

    data_handler.load_csv("data/processed_data/all_years.csv")
    december_data = pd.read_csv("data/processed_data/all_december.csv")

    return data_handler, december_data

def main():
    data_handler, december_data = preprocess_data()

    test_data_2023 = december_data[(december_data['Year'] == 2023) & (december_data['Month'] == 12)].copy()
    if test_data_2023.empty:
        print("No test data available for December 2023. Skipping evaluation.")
        return

    feature_columns = ['Month', 'Day_of_Week', 'Hour', 'Intermittent_Production', 'Constant_Production']
    target_column = 'Sold[MW]'

    direct_model = direct_prediction(data_handler, test_data_2023, feature_columns, target_column)
    consum_model, productie_model = component_prediction(data_handler, test_data_2023, feature_columns)
    bucket_model = bucket_prediction(data_handler, test_data_2023, feature_columns)

    print("=== Feature Importances ===")
    print("Feature importances (Approach 1):", direct_model.feature_importances())
    print("Feature importances (Approach 2 - Consum):", consum_model.feature_importances())
    print("Feature importances (Approach 2 - Productie):", productie_model.feature_importances())
    print("Feature importances (Approach 3 - Bucketing):", bucket_model.feature_importances())


if __name__ == "__main__":
    main()